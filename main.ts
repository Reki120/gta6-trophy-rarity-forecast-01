// gta6-trophy-rarity-forecast-01: forecast each GTA VI trophy's Exophase earn rate
// at the case's target as five quantiles, scored by pinball loss on the logit scale.
//
// Method: collapse the history to distinct states, anchor on each trophy's
// last non-null rate. A flat history (one distinct state) means Exophase has not
// refreshed since launch night, so the median is pushed down by a refresh prior:
// the newly tracked players hold fewer trophies, so rates fall when the page
// refreshes. A rich history blends three views in logit space: a counts model
// (power-law growth of players and of per-trophy earners), a direct trend of
// logit(rate) on log-time, and persistence at the last rate. The spread is a
// random-walk sigma estimated from the observed state-to-state logit moves,
// plus a jump term, widened asymmetrically when the median shift is clearly
// negative.
//
// Node 22, standard library only. Reads every <id>.in in the input directory,
// writes <id>.out to the output directory. Deterministic: same input, same bytes.

import { mkdirSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

const INPUT_DIR = process.env.MAKEFUN_INPUT_DIR ?? '/input';
const OUTPUT_DIR = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';

/** The quantile levels the contest scores, in the order each row lists them. */
const LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95];
/** Jump term in logit units: flat history / rich history. */
const JUMP_FLAT = 0.45;
const JUMP_RICH = 0.15;
/** Refresh prior: median logit shift, flat history only. */
const REFRESH_SHIFT = 2.0;
const REFRESH_TAU_HOURS = 1.2;
/** Default rate (percent) for a trophy with no rate anywhere in the history. */
const DEFAULT_RATE = 0.3;

interface Trophy {
  id: number;
}

interface Read {
  at: string;
  players_tracked: number | null;
  rates: Array<number | null>;
}

interface Case {
  id: string;
  at: string;
  target: string;
  horizon_minutes: number;
  levels: number[];
  trophies: Trophy[];
  history: Read[];
}

/** Ordinary least-squares slope of ys on xs. */
function olsSlope(xs: number[], ys: number[]): number {
  const n = xs.length;
  if (n < 2) return 0;
  let mx = 0;
  let my = 0;
  for (let i = 0; i < n; i++) {
    mx += xs[i];
    my += ys[i];
  }
  mx /= n;
  my /= n;
  let sxy = 0;
  let sxx = 0;
  for (let i = 0; i < n; i++) {
    sxy += (xs[i] - mx) * (ys[i] - my);
    sxx += (xs[i] - mx) * (xs[i] - mx);
  }
  return sxx > 0 ? sxy / sxx : 0;
}

/** Shrink a per-trophy slope toward the pooled median slope. */
function shrinkSlope(slope: number, pooled: number, nObs: number): number {
  const w = 2.5;
  return (nObs * slope + w * pooled) / (nObs + w);
}

const clampRate = (x: number): number => Math.min(99.99, Math.max(0.01, x));
const logit = (pct: number): number => {
  const r = clampRate(pct);
  return Math.log(r / (100 - r));
};
const invLogit = (x: number): number => 100 / (1 + Math.exp(-x));
/** x rounded to 4 decimals. */
const round4 = (x: number): number => Math.round(x * 10_000) / 10_000;
const clamp = (x: number, lo: number, hi: number): number => Math.min(hi, Math.max(lo, x));

/**
 * Inverse standard normal CDF (Acklam's rational approximation, ~1.15e-9
 * absolute error). Computed from the case's own levels so an unexpected level
 * list still yields quantiles instead of throwing.
 */
function normInv(p: number): number {
  const a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285984469534e+02, 1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00];
  const b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02, 6.680131188771972e+01, -1.328068155288572e+01];
  const c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00, -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00];
  const d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00, 3.754408661907416e+00];
  const pLow = 0.02425;
  const pHigh = 1 - pLow;
  let q: number;
  let r: number;
  if (p < pLow) {
    q = Math.sqrt(-2 * Math.log(p));
    return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) /
      ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1);
  }
  if (p > pHigh) {
    q = Math.sqrt(-2 * Math.log(1 - p));
    return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) /
      ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1);
  }
  q = p - 0.5;
  r = q * q;
  return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q /
    (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1);
}

function median(xs: number[]): number {
  if (xs.length === 0) return 0;
  const s = [...xs].sort((a, b) => a - b);
  const m = Math.floor(s.length / 2);
  return s.length % 2 === 1 ? s[m] : (s[m - 1] + s[m]) / 2;
}

/** One distinct state: consecutive reads with identical players and rates merge. */
interface State {
  atMs: number;
  players: number | null;
  rates: Array<number | null>;
}

function distinctStates(history: Read[]): State[] {
  const states: State[] = [];
  for (const read of history) {
    const atMs = Date.parse(read.at);
    const prev = states[states.length - 1];
    if (prev) {
      const samePlayers = prev.players === read.players_tracked;
      const sameRates = prev.rates.length === read.rates.length &&
        prev.rates.every((r, i) => r === read.rates[i]);
      if (samePlayers && sameRates) continue;
    }
    states.push({ atMs, players: read.players_tracked, rates: [...read.rates] });
  }
  return states;
}

/** Last non-null rate per trophy across the states, oldest to newest. */
function lastRates(states: State[], n: number): Array<number | null> {
  const out: Array<number | null> = new Array(n).fill(null);
  for (const s of states) {
    for (let i = 0; i < n; i++) {
      if (s.rates[i] !== null && s.rates[i] !== undefined) out[i] = s.rates[i];
    }
  }
  return out;
}

/**
 * Per-day logit volatility from the state-to-state moves: for every adjacent
 * state pair and every trophy with a rate in both, the squared logit
 * increment divided by the gap in days; the median over all of them, square
 * rooted. Zero when there is only one distinct state.
 */
function sigmaDay(states: State[], n: number): number {
  const terms: number[] = [];
  for (let k = 1; k < states.length; k++) {
    const dDays = (states[k].atMs - states[k - 1].atMs) / 86_400_000;
    if (!(dDays > 0)) continue;
    for (let i = 0; i < n; i++) {
      const a = states[k - 1].rates[i];
      const b = states[k].rates[i];
      if (a === null || a === undefined || b === null || b === undefined) continue;
      const d = logit(b) - logit(a);
      terms.push((d * d) / dDays);
    }
  }
  return Math.sqrt(median(terms));
}

function forecastCase(c: Case): { trophies: Array<{ id: number; q: number[] }> } {
  const n = c.trophies.length;
  const horizonDays = c.horizon_minutes / 1440;
  const horizonHours = c.horizon_minutes / 60;
  const z = c.levels.map((t) => normInv(clamp(t, 0.0001, 0.9999)));

  const states = distinctStates(c.history ?? []);
  const last = lastRates(states, n);
  const flat = states.length <= 1;
  const jump = flat ? JUMP_FLAT : JUMP_RICH;
  const sDay = flat ? 0 : sigmaDay(states, n);

  // --- Rich-history ensemble (step 3), fitted in x = ln(1 + hours since
  // the first read), anchored at the last state, extrapolated to the target.
  const K = states.length;
  const t0 = states.length > 0 ? states[0].atMs : Date.parse(c.at);
  const xOf = (ms: number): number => Math.log(1 + Math.max(0, ms - t0) / 3_600_000);
  const targetMs = c.target ? Date.parse(c.target) : Date.parse(c.at) + c.horizon_minutes * 60_000;
  const xTarget = xOf(targetMs);
  const xLast = states.length > 0 ? xOf(states[states.length - 1].atMs) : 0;
  const dxTarget = Math.max(0, xTarget - xLast);

  // (a) counts model: power-law slope of ln(players) on x.
  const playersXs: number[] = [];
  const playersYs: number[] = [];
  let lastPlayers: number | null = null;
  for (const s of states) {
    if (s.players !== null && s.players > 0) {
      playersXs.push(xOf(s.atMs));
      playersYs.push(Math.log(s.players));
      lastPlayers = s.players;
    }
  }
  const playersSlope = olsSlope(playersXs, playersYs);
  const logGrowthCap = Math.log(1 + 2 * horizonDays + 0.5);
  const logPlayersGrowth = Math.min(playersSlope * dxTarget, logGrowthCap);
  const playersTarget = lastPlayers !== null ? lastPlayers * Math.exp(logPlayersGrowth) : null;

  // Per-trophy slopes, each shrunk toward the pooled median slope.
  const earnerSlopes: number[] = [];
  const trendSlopes: number[] = [];
  const earnerData: Array<{ xs: number[]; ys: number[]; lastEarners: number | null }> = [];
  const trendData: Array<{ xs: number[]; ys: number[] }> = [];
  for (let i = 0; i < n; i++) {
    const xs: number[] = [];
    const ysLogit: number[] = [];
    const ysEarn: number[] = [];
    let lastEarn: number | null = null;
    for (const s of states) {
      const r = s.rates[i];
      if (r === null || r === undefined) continue;
      xs.push(xOf(s.atMs));
      ysLogit.push(logit(r));
      if (s.players !== null && s.players > 0) {
        const earners = (r * s.players) / 100;
        ysEarn.push(Math.log(earners + 0.5));
        lastEarn = earners;
      }
    }
    earnerSlopes.push(olsSlope(xs, ysEarn));
    trendSlopes.push(olsSlope(xs, ysLogit));
    earnerData.push({ xs, ys: ysEarn, lastEarners: lastEarn });
    trendData.push({ xs, ys: ysLogit });
  }
  const pooledEarnerSlope = median(earnerSlopes);
  const pooledTrendSlope = median(trendSlopes);

  // Blend weights by number of distinct states.
  let wCounts = 0;
  let wTrend = 0;
  let wPersist = 1;
  if (K >= 10) {
    [wCounts, wTrend, wPersist] = [0.45, 0.3, 0.25];
  } else if (K >= 5) {
    [wCounts, wTrend, wPersist] = [0.4, 0.25, 0.35];
  } else if (K >= 3) {
    [wCounts, wTrend, wPersist] = [0.3, 0.2, 0.5];
  } else if (K === 2) {
    [wCounts, wTrend, wPersist] = [0.2, 0.15, 0.65];
  }

  const rows: Array<{ id: number; q: number[] }> = [];
  for (let i = 0; i < n; i++) {
    const anchor = last[i] === null ? DEFAULT_RATE : last[i];
    const anchorLogit = logit(anchor);

    let medLogit = anchorLogit;
    if (flat) {
      // Flat history: the refresh prior (step 2).
      medLogit = anchorLogit - REFRESH_SHIFT * (1 - Math.exp(-horizonHours / REFRESH_TAU_HOURS));
    } else {
      // (a) counts model: predicted rate = 100 * earners / players at target.
      const ed = earnerData[i];
      const earnerSlope = shrinkSlope(earnerSlopes[i], pooledEarnerSlope, ed.xs.length);
      let logitCounts = anchorLogit;
      if (playersTarget !== null && playersTarget > 0) {
        const lastEarn = ed.lastEarners !== null ? ed.lastEarners : (anchor * (lastPlayers ?? 0)) / 100;
        const earnersTarget = Math.max(0, Math.exp(Math.log(lastEarn + 0.5) + earnerSlope * dxTarget) - 0.5);
        logitCounts = logit((100 * earnersTarget) / playersTarget);
      }
      // (b) direct trend: OLS slope of logit(rate) on x, shrunk, anchored.
      const td = trendData[i];
      const trendSlope = shrinkSlope(trendSlopes[i], pooledTrendSlope, td.xs.length);
      const logitTrend = anchorLogit + trendSlope * dxTarget;
      // (c) blend in logit space; K = 2 also takes half the flat prior shift.
      medLogit = wCounts * logitCounts + wTrend * logitTrend + wPersist * anchorLogit;
      if (K === 2) {
        medLogit -= 0.5 * REFRESH_SHIFT * (1 - Math.exp(-horizonHours / REFRESH_TAU_HOURS));
      }
    }
    medLogit = clamp(medLogit, anchorLogit - 1.6, anchorLogit + 1.6);

    // Spread: random-walk sigma over the horizon plus the jump term.
    const sigma = clamp(Math.sqrt(sDay * sDay * horizonDays + jump * jump), 0.3, 2.0);

    // Asymmetric widening when the median shift is clearly negative.
    const shift = medLogit - anchorLogit;
    const sigmaDown = shift < -0.25 ? sigma * 1.15 : sigma;
    const sigmaUp = shift < -0.25 ? sigma * 0.9 : sigma;

    const q = z.map((zj) => {
      const s = zj < 0 ? sigmaDown : sigmaUp;
      return round4(clampRate(invLogit(medLogit + zj * s)));
    });
    // Enforce non-decreasing after rounding.
    for (let j = 1; j < q.length; j++) {
      if (q[j] < q[j - 1]) q[j] = q[j - 1];
    }
    rows.push({ id: c.trophies[i].id, q });
  }
  return { trophies: rows };
}

mkdirSync(OUTPUT_DIR, { recursive: true });
const inputs = readdirSync(INPUT_DIR).filter((f) => f.endsWith('.in')).sort();
for (const file of inputs) {
  const id = file.slice(0, -'.in'.length);
  const c = JSON.parse(readFileSync(join(INPUT_DIR, file), 'utf8')) as Case;
  const out = forecastCase(c);
  writeFileSync(join(OUTPUT_DIR, `${id}.out`), JSON.stringify(out));
}
