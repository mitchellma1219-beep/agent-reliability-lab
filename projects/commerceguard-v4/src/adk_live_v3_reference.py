"""Optional Google ADK/Gemini four-agent debate, research-only.
Not authenticated or tested against a live Gemini account in this release.
pip install google-adk && export GOOGLE_API_KEY=...
"""
import asyncio
import json
import os
from pathlib import Path
from research import DataAnalyst,IndependentAuditor,ScenarioSimulator,load_weekly

ROOT=Path(__file__).resolve().parents[1]
GUARD=("All data are observational; never claim measured incrementality or profit. "
       "You may only propose research. You cannot execute or authorize ad spending. "
       "Return a single JSON object.")

def parse(raw):
    raw=raw.strip()
    if raw.startswith(chr(96)*3):
        raw="\n".join(raw.splitlines()[1:-1]).strip()
    obj=json.loads(raw)
    if not isinstance(obj,dict): raise ValueError("Expected JSON object")
    return obj

async def ask(name,instruction,payload,model):
    from google.adk.agents import LlmAgent
    from google.adk.apps import App
    from google.adk.runners import InMemoryRunner
    from google.genai import types
    agent=LlmAgent(name=name,model=model,instruction=GUARD+" "+instruction)
    app=App(name="cg_"+name.lower(),root_agent=agent)
    async with InMemoryRunner(app=app) as runner:
        session=await runner.session_service.create_session(app_name=app.name,user_id="researcher")
        pieces=[]
        async for event in runner.run_async(
            user_id="researcher",session_id=session.id,
            new_message=types.Content(role="user",parts=[types.Part.from_text(text=json.dumps(payload))])
        ):
            if event.is_final_response() and event.content:
                pieces.extend(p.text for p in event.content.parts if getattr(p,"text",None))
        if not pieces:raise RuntimeError("Agent returned no final response")
        return parse("".join(pieces))

async def deliberate(model="gemini-2.5-flash",target_week=52):
    if not os.getenv("GOOGLE_API_KEY"):raise EnvironmentError("GOOGLE_API_KEY is required")
    history=[w for w in load_weekly(ROOT/"data/public_weekly_2019.csv") if w.week<target_week]
    a=DataAnalyst().run(history,target_week)
    analysis=await ask("DataAnalyst","Inspect attribution/data reliability. Keys: risks,observations.",{"analysis":a},model)
    attempts=[];feedback=[]
    for i in range(5):
        proposal=await ask("StrategyPlanner",
            "Propose research-only budget shift with keys recipient (facebook or google), "
            "transfer_fraction (number between 0 and 0.1) and reason.",
            {"analysis":a,"review":analysis,"feedback":feedback},model)
        p={k:proposal.get(k) for k in ("recipient","transfer_fraction","reason")}
        try:sim=ScenarioSimulator().stress(history,p)
        except (ValueError,KeyError,TypeError,OverflowError):sim=None
        critique=await ask("SimulationCritic",
            "Critique the simulated assumptions, without changing numbers. Keys: critique,limitations.",
            {"proposal":p,"simulation":sim},model)
        auditor=await ask("IndependentAuditor",
            "Find attribution and safety risks. You cannot authorize deployment. Keys: concerns,recommendation.",
            {"proposal":p,"simulation":sim,"critique":critique,"analysis":a},model)
        gate=IndependentAuditor().evaluate(a,p,sim,history)
        attempts.append({"proposal":p,"simulation":sim,"critic":critique,"llm_auditor":auditor,"hard_gate":gate})
        if gate["research_pass"]:break
        feedback.append({"violations":gate["violations"]})
    result={"llm_review":analysis,"attempts":attempts,"production_write_allowed":False,
            "release_status":"RESEARCH_ONLY_HUMAN_REVIEW"}
    output=ROOT/"outputs/llm_proposals_research_only.json"
    output.parent.mkdir(exist_ok=True)
    output.write_text(json.dumps(result,indent=2))
    return result

if __name__=="__main__":asyncio.run(deliberate())
