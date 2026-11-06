"""Semi-synthetic audit data from the RetailRocket e-commerce dataset.

Real part:   the interaction graph. Each audit day t uses the typed (visitor, item) edges
             observed in the 7-day window [t-6, t]; an edge's type is the strongest event
             in the window (view < add-to-cart < transaction).
Pre-trained: item embeddings x (16-d) from an SVD of days 0-29, which are excluded from
             every audit window, playing the role of the platform's intrinsic item embeddings.
Simulated:   opt-out status. Each seed opts out a random 10% of customers and lets status
             switch daily (in->out 0.005, out->in 0.045), as in the synthetic simulator.
Cleaning:    visitors with fewer than 2 or more than 200 events in total are dropped
             (single-event visitors carry no co-interaction; >200 are mostly crawlers).

Usage: import real_data; w = real_data.RealWorld(seed); then use like audit_sim.World.
`load()` must be called once; it sets audit_sim.U / audit_sim.I to the real sizes.
Data: run fetch_retailrocket.sh first (downloads events.csv into data/retailrocket/).
"""
import os
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.utils.extmath import randomized_svd
import audit_sim as a

HERE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(HERE, "..", "data", "retailrocket", "events.csv")
CACHE = os.path.join(HERE, "..", "data", "retailrocket", "processed.npz")
WINDOW = 7
PRETRAIN_DAYS = 30
FIRST_AUDIT_DAY = PRETRAIN_DAYS + WINDOW      # first window lies entirely after pre-training
TYPE = {"view": 0, "addtocart": 1, "transaction": 2}

_DATA = None


def _preprocess():
    df = pd.read_csv(RAW, usecols=["timestamp", "visitorid", "event", "itemid"])
    df["day"] = (df.timestamp // 86_400_000 - df.timestamp.min() // 86_400_000).astype(np.int32)
    cnt = df.groupby("visitorid").size()
    keep = cnt[(cnt >= 2) & (cnt <= 200)].index
    df = df[df.visitorid.isin(keep)]
    df["u"] = pd.factorize(df.visitorid)[0].astype(np.int32)
    df["i"] = pd.factorize(df.itemid)[0].astype(np.int32)
    df["t"] = df.event.map(TYPE).astype(np.int8)
    np.savez_compressed(CACHE, u=df.u.values, i=df.i.values, t=df.t.values, day=df.day.values)


def load():
    global _DATA
    if _DATA is None:
        if not os.path.exists(CACHE):
            _preprocess()
        z = np.load(CACHE)
        u, i, t, day = z["u"], z["i"], z["t"], z["day"]
        U, I = int(u.max()) + 1, int(i.max()) + 1
        # pre-trained item embeddings from days < PRETRAIN_DAYS
        m = day < PRETRAIN_DAYS
        A = sp.csr_matrix((np.ones(m.sum()), (u[m], i[m])), shape=(U, I))
        A.data[:] = 1
        _, S, Vt = randomized_svd(A, a.D_X, n_iter=4, random_state=0)
        x = Vt.T * np.sqrt(S)
        x = x / (np.linalg.norm(x, axis=1, keepdims=True).mean() + 1e-12)
        unseen = np.asarray(A.sum(0)).ravel() == 0
        x[unseen] = np.random.default_rng(0).normal(scale=0.05, size=(unseen.sum(), a.D_X))
        order = np.argsort(day, kind="stable")
        _DATA = dict(u=u[order], i=i[order], t=t[order], day=day[order], U=U, I=I, x=x,
                     last_day=int(day.max()))
    a.U, a.I = _DATA["U"], _DATA["I"]
    return _DATA


class RealWorld:
    """Same interface as audit_sim.World: snapshot(), step(), .x, .optout, .w."""

    def __init__(self, seed, start_day=FIRST_AUDIT_DAY):
        d = load()
        self.d = d
        self.rng = np.random.default_rng(seed)
        self.x = d["x"]
        self.day = start_day
        self.optout = self.rng.random(d["U"]) < 0.10
        self.w = np.ones(d["I"])                      # unused; kept for interface parity
        self._idx = np.searchsorted(d["day"], np.arange(d["last_day"] + 2))

    def _window(self, t):
        lo, hi = self._idx[max(t - WINDOW + 1, 0)], self._idx[t + 1]
        u, i, ty, dd = self.d["u"][lo:hi], self.d["i"][lo:hi], self.d["t"][lo:hi], self.d["day"][lo:hi]
        key = u.astype(np.int64) * self.d["I"] + i
        # one edge per (visitor, item): strongest type in the window; eday = last interaction day
        order = np.lexsort((-ty, key))
        key, ty, dd = key[order], ty[order], dd[order]
        first = np.r_[True, key[1:] != key[:-1]]
        starts = np.flatnonzero(first)
        eday = np.maximum.reduceat(dd, starts)
        key, ty = key[first], ty[first]
        return ((key // self.d["I"]).astype(np.int64), (key % self.d["I"]).astype(np.int64),
                ty.astype(np.int64), eday.astype(np.int64))

    def snapshot(self):
        u, i, t, eday = self._window(self.day)
        return dict(u=u, i=i, t=t, out=self.optout[u], optout=self.optout.copy(), x=self.x,
                    day=self.day, eday=eday)

    def step(self):
        flip = np.where(self.optout, self.rng.random(self.d["U"]) < a.P_OUT2IN,
                        self.rng.random(self.d["U"]) < a.P_IN2OUT)
        self.optout = self.optout ^ flip
        self.day += 1
