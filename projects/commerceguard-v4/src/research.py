"""CommerceGuard V4: four-role read-only marketing research pipeline (stdlib only).

Observational platform conversion rates are NOT marginal causal returns.
No Google Ads or Meta API connections, writes or autonomous spending.
"""
from __future__ import annotations
import csv
import hashlib
import json
import math
import statistics
from dataclasses import dataclass
from pathlib import Path

CHANNELS = ("facebook", "google")

@dataclass(frozen=True)
class Week:
    week: int
    spends: dict
    conversions: dict


def load_weekly(path: Path) -> list[Week]:
    rows = []
    with path.open(newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if int(r["days"]) != 7:
                continue
            spends = {"facebook": float(r["fb_spend"]), "google": float(r["google_spend"])}
            conversions = {"facebook": float(r["fb_conversions"]), "google": float(r["google_conversions"])}
            if any(not math.isfinite(v) or v <= 0 for v in spends.values()):
                raise ValueError("nonpositive or invalid spend")
            if any(not math.isfinite(v) or v < 0 for v in conversions.values()):
                raise ValueError("invalid conversions")
            rows.append(Week(int(r["week"]), spends, conversions))
    rows.sort(key=lambda x: x.week)
    if len(rows) < 20 or len({w.week for w in rows}) != len(rows):
        raise ValueError("incomplete or duplicated weekly data")
    if any(b.week != a.week + 1 for a, b in zip(rows, rows[1:])):
        raise ValueError("time gap")
    return rows


def rate(w: Week, channel: str) -> float:
    return w.conversions[channel] / w.spends[channel]


class DataAnalyst:
    def run(self, past: list[Week], target: int, window=12, prior_weight=4.0):
        if len(past) < 12 or past[-1].week >= target:
            raise ValueError("lookahead or insufficient history")
        if not 2 <= window <= len(past):
            raise ValueError("invalid lookback")
        estimates = {}
        for ch in CHANNELS:
            recent = past[-window:]
            recent_rate = sum(w.conversions[ch] for w in recent) / sum(w.spends[ch] for w in recent)
            base_rate = sum(w.conversions[ch] for w in past) / sum(w.spends[ch] for w in past)
            estimates[ch] = (window*recent_rate + prior_weight*base_rate)/(window+prior_weight)
        return {"as_of":past[-1].week,"target":target,"rates":estimates,"window":window,
                "warning":"Cross-platform reported conversions may overlap; not causal."}


class StrategyPlanner:
    def propose(self, analysis: dict, attempt: int):
        recipient = max(CHANNELS, key=lambda c: analysis["rates"][c])
        return {"recipient":recipient,"transfer_fraction":[.20,.15,.10,.05,0][min(attempt,4)],
                "reason":"Observational efficiency only; requires experimentation."}


class ScenarioSimulator:
    @staticmethod
    def budgets(base: dict, policy: dict):
        recipient=policy.get("recipient")
        f=policy.get("transfer_fraction")
        if recipient not in CHANNELS or isinstance(f,bool) or not isinstance(f,(float,int)) or not math.isfinite(f):
            raise ValueError("invalid policy")
        if not 0 <= f <= 1: raise ValueError("invalid transfer")
        donor="google" if recipient=="facebook" else "facebook"
        shift=base[recipient]*f
        result=dict(base)
        result[recipient]+=shift
        result[donor]-=shift
        if result[donor]<0: raise ValueError("negative budget")
        return result

    def stress(self, history: list[Week], policy: dict, elasticities=(.3,.6,1.0)):
        result={}
        for elasticity in elasticities:
            gains=[]
            for w in history[-4:]:
                new=self.budgets(w.spends,policy)
                hypothetical=sum(w.conversions[c]*(new[c]/w.spends[c])**elasticity for c in CHANNELS)
                gains.append(hypothetical-sum(w.conversions.values()))
            result[str(elasticity)]={"assumption_based_gain":sum(gains),"worst_week":min(gains)}
        return result


class IndependentAuditor:
    def evaluate(self, analysis, proposal, simulation, history, max_step=.1):
        reasons=[]
        if analysis["as_of"]!=history[-1].week or analysis["target"]<=history[-1].week:
            reasons.append("future_data_leak")
        try:
            fraction=proposal["transfer_fraction"]
            if isinstance(fraction,bool) or not isinstance(fraction,(int,float)) or not math.isfinite(fraction):
                reasons.append("invalid_fraction")
            elif not 0 <= fraction <= max_step + 1e-12:
                reasons.append("over_limit")
            base={c:sum(w.spends[c] for w in history[-4:])/4 for c in CHANNELS}
            new=ScenarioSimulator.budgets(base,proposal)
            if abs(sum(base.values())-sum(new.values()))>1e-7*sum(base.values()):
                reasons.append("budget_not_conserved")
        except (ValueError,TypeError,KeyError,OverflowError):
            reasons.append("invalid_policy")
        if simulation is None:
            reasons.append("simulation_failed")
        return {"research_pass":not reasons,"violations":reasons,
                "production_write_allowed":False,"human_approval_required":True}


class Orchestrator:
    def decide(self, history: list[Week], target: int, window=12):
        analysis=DataAnalyst().run(history,target,window)
        attempts=[]
        for i in range(5):
            policy=StrategyPlanner().propose(analysis,i)
            try:
                stress=ScenarioSimulator().stress(history,policy)
            except (KeyError,TypeError,ValueError,OverflowError):
                stress=None
            audit=IndependentAuditor().evaluate(analysis,policy,stress,history)
            attempts.append({"round":i+1,"policy":policy,"stress":stress,"audit":audit})
            if audit["research_pass"]:
                return {"decision":"research_only","policy":policy,"attempts":attempts,
                        "production_write_allowed":False}
        return {"decision":"rejected","policy":None,"attempts":attempts,
                "production_write_allowed":False}


def evaluate(rows: list[Week], min_train=12, windows=(4,8,12,20)):
    if len(rows)<=min_train+10: raise ValueError("insufficient holdout")
    predictions=[]
    decisions=[]
    for i in range(min_train,len(rows)):
        past=rows[:i]
        future=rows[i] # evaluation ONLY, never sent to policy inputs
        eligible=[w for w in windows if w<=len(past)]
        inner_scores={}
        for window in eligible:
            errors=[]
            for j in range(max(12,i-16),i):
                if j<window: continue
                pred=DataAnalyst().run(rows[:j],rows[j].week,window)["rates"]
                errors.append(sum(abs(pred[c]-rate(rows[j],c)) for c in CHANNELS)/2)
            inner_scores[window]=statistics.mean(errors) if errors else float("inf")
        selected=min(eligible,key=lambda w:(inner_scores[w],w))
        models={
            "last_week":{ch:rate(past[-1],ch) for ch in CHANNELS},
            "trailing_12":DataAnalyst().run(past,future.week,12,0)["rates"],
            "nested_regularized":DataAnalyst().run(past,future.week,selected)["rates"],
        }
        for model,estimate in models.items():
            for ch in CHANNELS:
                predictions.append({"week":future.week,"model":model,"channel":ch,
                                    "selected_window":selected,"forecast":estimate[ch],
                                    "actual":rate(future,ch),
                                    "abs_error":abs(estimate[ch]-rate(future,ch))})
        proposal=Orchestrator().decide(past,future.week,selected)
        decisions.append({"week":future.week,"history_end":past[-1].week,
                          "selected_window":selected,"status":proposal["decision"],
                          "rounds":len(proposal["attempts"]),
                          "recipient":proposal["policy"]["recipient"] if proposal["policy"] else "",
                          "fraction":proposal["policy"]["transfer_fraction"] if proposal["policy"] else "",
                          "live_execution_allowed":False})
    metrics={ch:{m:statistics.mean(p["abs_error"] for p in predictions
                 if p["channel"]==ch and p["model"]==m)
                 for m in ("last_week","trailing_12","nested_regularized")} for ch in CHANNELS}
    return {"data_type":"third_party_public_observational_unverified",
            "holdout_weeks":len(decisions),"forecast_mae":metrics,
            "research_passes":sum(d["status"]=="research_only" for d in decisions),
            "live_account_changes":0,"predictions":predictions,"decisions":decisions}


def save(root: Path):
    source=root/"data/public_weekly_2019.csv"
    report=evaluate(load_weekly(source))
    folder=root/"outputs";folder.mkdir(exist_ok=True)
    for filename,key in (("forecasts_v4.csv","predictions"),("policies_v4.csv","decisions")):
        with (folder/filename).open("w",newline="",encoding="utf-8") as f:
            writer=csv.DictWriter(f,fieldnames=list(report[key][0]))
            writer.writeheader();writer.writerows(report[key])
    summary={k:v for k,v in report.items() if k not in ("predictions","decisions")}
    summary["source_sha256"]=hashlib.sha256(source.read_bytes()).hexdigest()
    (folder/"backtest_v4.json").write_text(json.dumps(summary,indent=2))
    return summary


if __name__=="__main__":
    print(json.dumps(save(Path(__file__).resolve().parents[1]),indent=2))
