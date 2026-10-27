"""Train and evaluate the supervised user-response model on real KuaiRec logs, with simple baselines.

Evaluation (all on data the model never trained on):
  1. temporal hold-out: target rows in the test window (2020-09-01..05) for seen / small / unseen users
  2. fully-observed matrix: rank all 3,327 catalog items for each small-matrix user from the history available at
     the end of the train window; ground truth = that user's actual watch ratios (no exposure bias).

Usage: python scripts/train_response_model.py --config configs/response_model.yaml --seed 0
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import torch
import torch.nn.functional as F

from seqrec_rl import PROCESSED_DIR
from seqrec_rl.data.prepare import load_splits
from seqrec_rl.data.sequences import SequenceExamples, load_interactions
from seqrec_rl.evaluation import metrics as M
from seqrec_rl.training.runs import Run, limit_threads, load_config, set_seed
from seqrec_rl.user_models.response_model import ResponseModel


def predict(model, ex, rows, dev, bs):
    wr, eng = [], []
    model.eval()
    with torch.no_grad():
        for s in range(0, len(rows), bs):
            b = ex.batch(rows[s:s + bs])
            w, e = model(torch.as_tensor(b["hist_items"], device=dev), torch.as_tensor(b["hist_rewards"], device=dev),
                         torch.as_tensor(b["target_item"], device=dev))
            wr.append(w.clamp(0, ex.wr.max()).cpu().numpy())
            eng.append(torch.sigmoid(e).cpu().numpy())
    return np.concatenate(wr), np.concatenate(eng)


def baselines(ex, train_rows, n_items, shrink=20.0):
    y = ex.wr[train_rows]
    e = (y >= 1.0).astype(np.float64)
    it = ex.item[train_rows]
    g, ge = y.mean(), e.mean()
    cnt = np.bincount(it, minlength=n_items)
    item_wr = (np.bincount(it, weights=y, minlength=n_items) + shrink * g) / (cnt + shrink)
    item_e = (np.bincount(it, weights=e, minlength=n_items) + shrink * ge) / (cnt + shrink)
    users = ex.user[train_rows]
    uniq, uinv = np.unique(users, return_inverse=True)
    res = y - item_wr[it]
    ucnt = np.bincount(uinv)
    ub = np.bincount(uinv, weights=res) / (ucnt + shrink)
    resl = e - item_e[it]
    ube = np.bincount(uinv, weights=resl) / (ucnt + shrink)
    return {"global": (g, ge), "item_wr": item_wr, "item_e": item_e,
            "user_bias": dict(zip(uniq, ub)), "user_bias_e": dict(zip(uniq, ube))}


def eval_rows(name, ex, rows, wr_hat, eng_hat):
    y = ex.wr[rows]
    yb = (y >= 1.0).astype(float)
    e, table = M.ece(yb, eng_hat)
    return {
        "set": name, "n": int(len(rows)),
        "rmse_wr": M.rmse(y, wr_hat), "mae_wr": M.mae(y, wr_hat),
        "auc_eng": M.auc(yb, eng_hat), "gauc_eng": M.gauc(ex.user[rows], yb, eng_hat),
        "logloss_eng": M.logloss(yb, eng_hat), "ece_eng": e, "reliability_eng": table,
        "wr_calibration": M.regression_calibration(y, wr_hat),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/response_model.yaml")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--name", default="response_model")
    ap.add_argument("overrides", nargs="*")
    args = ap.parse_args()
    cfg = load_config(args.config, args.overrides)
    limit_threads(cfg["threads"])
    set_seed(args.seed)
    run = Run(args.name, args.seed, cfg)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    splits = load_splits()
    n_all, n_cat = splits["n_items_all"], splits["n_catalog"]
    inter = load_interactions()
    ex = SequenceExamples(inter, cfg["hist_len"], pad=n_all, wr_cap=cfg["wr_cap"])
    tr = ex.rows("train", tuple(cfg["train_groups"]))
    va = ex.rows("val", ("seen", "small"))
    rng = np.random.default_rng(args.seed)
    if cfg["max_train_rows"]:
        tr = rng.choice(tr, cfg["max_train_rows"], replace=False)
    va_sub = rng.choice(va, min(len(va), 300_000), replace=False)

    model = ResponseModel(n_all, cfg["emb_dim"], cfg["hidden"], cfg["wr_cap"]).to(dev)
    y_tr = ex.wr[tr]
    e_mean = float((y_tr >= 1.0).mean())
    with torch.no_grad():  # start from the training-set means instead of 0
        model.glob.copy_(torch.tensor([float(y_tr.mean()), np.log(e_mean / (1 - e_mean))]))
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    best, best_state = np.inf, None
    for epoch in range(cfg["epochs"]):
        model.train()
        perm = rng.permutation(tr)
        tot, nb = 0.0, 0
        for s in range(0, len(perm), cfg["batch_size"]):
            b = ex.batch(perm[s:s + cfg["batch_size"]])
            hi = torch.as_tensor(b["hist_items"], device=dev)
            hr = torch.as_tensor(b["hist_rewards"], device=dev)
            ti = torch.as_tensor(b["target_item"], device=dev)
            y = torch.as_tensor(b["target_wr"], device=dev)
            wr, eng = model(hi, hr, ti)
            loss = F.huber_loss(wr, y, delta=cfg["huber_delta"]) + cfg["eng_loss_weight"] * \
                F.binary_cross_entropy_with_logits(eng, (y >= 1.0).float())
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            tot += loss.item()
            nb += 1
            if nb % 500 == 0:
                run.log(epoch=epoch, step=nb, train_loss=tot / nb)
        wr_hat, eng_hat = predict(model, ex, va_sub, dev, cfg["eval_batch"])
        vm = eval_rows("val", ex, va_sub, wr_hat, eng_hat)
        score = vm["rmse_wr"] + vm["logloss_eng"]
        run.log(epoch=epoch, train_loss=tot / nb, val_rmse=vm["rmse_wr"], val_auc=vm["auc_eng"],
                val_logloss=vm["logloss_eng"])
        print(f"epoch {epoch} loss {tot/nb:.4f} val rmse {vm['rmse_wr']:.4f} auc {vm['auc_eng']:.4f}")
        if score < best:
            best, best_state = score, {k: v.detach().clone() for k, v in model.state_dict().items()}
            torch.save(best_state, run.dir / "model.pt")
    model.load_state_dict(best_state)

    # ---------------- temporal hold-out -----------------------------------------------------------
    bl = baselines(ex, tr, n_all)
    results = {"temporal_test": [], "fo_matrix": {}}
    for group in ("seen", "small", "unseen"):
        rows = ex.rows("test", (group,))
        wr_hat, eng_hat = predict(model, ex, rows, dev, cfg["eval_batch"])
        r = eval_rows(f"test/{group}/gru_two_tower", ex, rows, wr_hat, eng_hat)
        results["temporal_test"].append(r)
        it, us = ex.item[rows], ex.user[rows]
        g, ge = bl["global"]
        results["temporal_test"].append(eval_rows(f"test/{group}/global_mean", ex, rows,
                                                  np.full(len(rows), g), np.full(len(rows), ge)))
        results["temporal_test"].append(eval_rows(f"test/{group}/item_mean", ex, rows, bl["item_wr"][it],
                                                  np.clip(bl["item_e"][it], 1e-4, 1 - 1e-4)))
        ub = np.array([bl["user_bias"].get(u, 0.0) for u in us])
        ube = np.array([bl["user_bias_e"].get(u, 0.0) for u in us])
        results["temporal_test"].append(eval_rows(f"test/{group}/user_item_bias", ex, rows,
                                                  np.clip(bl["item_wr"][it] + ub, 0, cfg["wr_cap"]),
                                                  np.clip(bl["item_e"][it] + ube, 1e-4, 1 - 1e-4)))
        for r in results["temporal_test"][-4:]:
            print(f"{r['set']:40s} rmse {r['rmse_wr']:.4f} auc {r['auc_eng']:.4f} gauc {r['gauc_eng']:.4f} "
                  f"ll {r['logloss_eng']:.4f} ece {r['ece_eng']:.4f}")

    # ---------------- fully-observed matrix ranking ------------------------------------------------
    W = np.load(PROCESSED_DIR / "fo_watch_ratio.npy")
    Wc = np.minimum(W, cfg["wr_cap"])
    users = np.array(splits["small_users_in_fo_order"])
    hist = ex.last_history(users, cfg["fo_history_cutoff_utc"])
    with torch.no_grad():
        model.eval()
        pred_wr, pred_eng = [], []
        for s in range(0, len(users), 256):
            w, e = model.score_all(torch.as_tensor(hist["hist_items"][s:s + 256], device=dev),
                                   torch.as_tensor(hist["hist_rewards"][s:s + 256], device=dev), n_cat)
            pred_wr.append(w.cpu().numpy())
            pred_eng.append(e.cpu().numpy())
    pred_wr, pred_eng = np.concatenate(pred_wr), np.concatenate(pred_eng)
    np.save(run.dir / "fo_pred_wr.npy", pred_wr.astype(np.float32))
    np.save(run.dir / "fo_pred_eng.npy", pred_eng.astype(np.float32))
    item_scores = np.broadcast_to(bl["item_wr"][:n_cat], W.shape)
    rng_scores = rng.random(W.shape)
    env_users = splits["env_users"]
    for subset in ("test", "all"):
        sel = np.isin(users, env_users["test"]) if subset == "test" else np.ones(len(users), bool)
        out = {}
        for name, sc in (("gru_two_tower", pred_wr), ("item_mean", item_scores), ("random", rng_scores)):
            r = {f"ndcg@{k}": float(M.ndcg_at_k(Wc[sel], sc[sel], k).mean()) for k in (10, 50)}
            top10 = np.argsort(-np.where(np.isnan(W[sel]), -np.inf, sc[sel]), axis=1)[:, :10]
            r["mean_true_wr_top10"] = float(np.nanmean(np.take_along_axis(Wc[sel], top10, 1)))
            r["spearman"] = float(np.nanmean(M.spearman_rows(Wc[sel], sc[sel])))
            out[name] = r
        m = ~np.isnan(W[sel])
        yb = (W[sel][m] >= 1.0).astype(float)
        out["gru_two_tower"].update({
            "rmse_wr": M.rmse(Wc[sel][m], pred_wr[sel][m]),
            "auc_eng": M.auc(yb, pred_eng[sel][m]),
            "ece_eng": M.ece(yb, pred_eng[sel][m])[0],
            "mean_pred_wr": float(pred_wr[sel][m].mean()), "mean_true_wr": float(Wc[sel][m].mean()),
        })
        oracle = np.where(np.isnan(W[sel]), -np.inf, Wc[sel])
        out["oracle_mean_true_wr_top10"] = float(np.mean(-np.sort(-oracle, axis=1)[:, :10]))
        results["fo_matrix"][subset] = out
        print(subset, json.dumps(out, indent=1))
    run.save_metrics(results)


if __name__ == "__main__":
    main()
