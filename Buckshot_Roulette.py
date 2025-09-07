import os, json, random, re, unicodedata
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional

import gradio as gr
from openai import OpenAI
from dotenv import load_dotenv

# ---------------------- 환경설정 ----------------------
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
OPENAI_MODEL = "gpt-4o-mini"

MAX_HEALTH = 4
BASE_DAMAGE = 1

ACTION_MAP = {
    "수갑": "use_cuffs",
    "돋보기": "use_peek",
    "담배": "use_heal",
    "나에게 쏘기": "shoot_self",
    "상대에게 쏘기": "shoot_opponent",
}

# ---------------------- 데이터 구조 ----------------------
@dataclass
class Item:
    key: str
    name: str

ITEMS_CATALOG = {
    "peek": Item("peek", "🔍 돋보기"),
    "cuffs": Item("cuffs", "🔗 수갑"),
    "heal": Item("heal", "🚬 담배"),
}

@dataclass
class GameState:
    round: int
    turn: str
    log: List[str]
    human_hp: int
    ai_hp: int
    magazine: List[str]
    known_next: Optional[str]
    human_items: Dict[str, int]
    ai_items: Dict[str, int]
    last_action: Optional[str]
    cuffed: Optional[str]

# ---------------------- 유틸 ----------------------
def pretty_items(items: Dict[str,int]) -> str:
    return ", ".join([f"{ITEMS_CATALOG[k].name} x{v}" for k,v in items.items() if v>0]) or "없음"

def has_final_consonant(kor_word: str) -> bool:
    # 마지막 글자 한글 여부 체크 후 종성 유무 판단
    if not kor_word:
        return False
    ch = kor_word[-1]
    code = ord(ch)
    if 0xAC00 <= code <= 0xD7A3:
        jong = (code - 0xAC00) % 28
        return jong != 0
    # 한글이 아니면 대충 '는' 처리
    return False

def topic_particle(label: str) -> str:  # '은/는'
    return "은" if has_final_consonant(label) else "는"

def subject_particle(actor: str) -> str:  # '이/가' (사격 로그용)
    # '당신(받침 있음: ㄴ)→이', '딜러(받침 없음)→가' 맞춤
    lab = actor_label(actor)
    return "이" if has_final_consonant(lab) else "가"

def actor_label(actor: str) -> str:
    return "당신" if actor=="human" else "딜러"

def draw_shell(state: GameState) -> str:
    shell = state.magazine.pop(0)
    state.known_next = None
    return shell

def damage(hp:int, amt:int=1): return max(0, hp-amt)
def heal(hp:int, amt:int=1): return min(MAX_HEALTH, hp+amt)

def check_end(state: GameState):
    if state.human_hp<=0 and state.ai_hp<=0: return "무승부"
    if state.human_hp<=0: return "패배"
    if state.ai_hp<=0: return "승리"
    if not state.magazine:
        if state.human_hp>state.ai_hp: return "승리 (체력 우세)"
        if state.human_hp<state.ai_hp: return "패배 (체력 열세)"
        return "무승부 (체력 동일)"
    return None

def render_log(state: GameState) -> str:
    return "\n".join(state.log)

# ---------------------- 라운드 생성 ----------------------
def new_round(prev:Optional[GameState]=None):
    """
    매 라운드:
    - HP 전부 4로 리셋
    - 플레이어 선공
    - 탄창/아이템 재지급
    """
    rnd = 1 if prev is None else prev.round+1
    total, live = random.randint(6,8), random.randint(2,4)
    blanks = total-live
    mag = ["live"]*live+["blank"]*blanks
    random.shuffle(mag)
    def roll(): return {k:(1 if random.random()<0.45 else 0) for k in ITEMS_CATALOG}

    state = GameState(
        round=rnd,
        turn="human",                       # 라운드 시작은 항상 플레이어
        log=[f"🎲 라운드 {rnd} 시작! 총 {total}발 (실탄 {live}, 공탄 {blanks})"],
        human_hp=MAX_HEALTH,                # 매 라운드 HP 리셋
        ai_hp=MAX_HEALTH,
        magazine=mag, known_next=None,
        human_items=roll(), ai_items=roll(),
        last_action=None,
        cuffed=None
    )
    # 시작 안내
    state.log.append(f"🎁 당신 아이템: {pretty_items(state.human_items)}")
    state.log.append(f"🎁 딜러 아이템: {pretty_items(state.ai_items)}")
    return state

# ---------------------- AI ----------------------
AI_SYSTEM_PROMPT = """당신은 이 게임의 딜러 AI입니다.
다음 중 하나만 JSON으로 답하세요:
["shoot_self","shoot_opponent","use_peek","use_cuffs","use_heal"]
형식:
{"action":"...", "reason":"한국어 간단 설명"}"""

CHAT_SYSTEM_PROMPT = """당신은 게임 딜러입니다. 플레이어와 대화하되, 비밀 정보(딜러가 peek으로 본 탄 정보)는 절대 누설하지 마세요."""

def call_openai(system:str,user:str):
    r = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[{"role":"system","content":system},{"role":"user","content":user}],
        temperature=0.3
    )
    return r.choices[0].message.content.strip()

def decide_ai_action(state:GameState):
    obs=json.dumps({
        "round":state.round,"ai_hp":state.ai_hp,"human_hp":state.human_hp,
        "remain":len(state.magazine),"known_next":state.known_next,
        "items":state.ai_items
    },ensure_ascii=False)
    raw=call_openai(AI_SYSTEM_PROMPT,obs)
    m=re.search(r'\{.*\}',raw,re.S)
    if m:
        try:
            d=json.loads(m.group(0)); return d.get("action","shoot_opponent"), d.get("reason","")
        except: pass
    return "shoot_opponent","파싱 실패"

def dealer_chat(state:GameState,history:list,user_msg:str):
    public=f"라운드 {state.round}, 당신 HP {state.human_hp}, 딜러 HP {state.ai_hp}, 탄 {len(state.magazine)}발"
    conv="".join([f"\n[플레이어] {a}\n[딜러] {b}" for a,b in history[-5:]])
    prompt=f"{public}\n{conv}\n[플레이어] {user_msg}\n[딜러]"
    return call_openai(CHAT_SYSTEM_PROMPT,prompt)

# ---------------------- 턴 계산(수갑 포함) ----------------------
def resolve_next_turn(state: GameState, actor: str, keep: bool) -> None:
    opp = "ai" if actor=="human" else "human"
    if keep:
        state.turn = actor
        return
    next_turn = opp
    if state.cuffed == opp:
        lab = actor_label(opp)
        state.log.append(f"⛓️ {lab}{topic_particle(lab)} 수갑으로 턴을 건너뜀")
        state.cuffed = None
        next_turn = actor
    state.turn = next_turn

# ---------------------- 액션 ----------------------
def apply_action(state:GameState,actor:str,action:str):
    opp="ai" if actor=="human" else "human"
    keep=False  # 기본은 턴 넘김. 조건에 따라 유지.

    def have(side,k): return (state.human_items if side=="human" else state.ai_items)[k]>0
    def consume(side,k):
        bag = state.human_items if side=="human" else state.ai_items
        bag[k] = max(0, bag[k]-1)

    if action=="use_peek":
        if have(actor,"peek"):
            state.known_next=state.magazine[0] if state.magazine else None
            consume(actor,"peek")
            state.log.append("🧪 당신 돋보기 사용" if actor=="human" else "🧪 딜러 돋보기 사용 (결과 비공개)")
            keep = True  # ★ 아이템은 무료 액션 → 턴 유지
        else:
            state.log.append("⚠️ 돋보기 없음")
            keep = True

    elif action=="use_cuffs":
        if have(actor,"cuffs"):
            if state.cuffed is None:
                consume(actor,"cuffs")
                state.cuffed = opp  # 상대의 '다음' 턴 1회 스킵 예약
                state.log.append(f"⛓️ {actor_label(actor)} 수갑 사용")
                keep = True  # ★ 아이템은 무료 액션 → 턴 유지
            else:
                state.log.append("⚠️ 이미 수갑 효과가 대기중")
                keep = True
        else:
            state.log.append("⚠️ 수갑 없음")
            keep = True

    elif action=="use_heal":
        if have(actor,"heal"):
            if actor=="human":
                before,after=state.human_hp,heal(state.human_hp)
                state.human_hp=after
            else:
                before,after=state.ai_hp,heal(state.ai_hp)
                state.ai_hp=after
            consume(actor,"heal")
            state.log.append(f"🚬 {actor_label(actor)} 체력 {before}→{after}")
            keep = True  # ★ 아이템은 무료 액션 → 턴 유지
        else:
            state.log.append("⚠️ 담배 없음")
            keep = True

    elif action in("shoot_self","shoot_opponent"):
        if not state.magazine:
            state.log.append("⚠️ 탄 없음")
            keep = True
        else:
            s=draw_shell(state)
            tgt=actor if action=="shoot_self" else opp
            hit=(s=="live")
            part=subject_particle(actor)

            if hit:
                if tgt=="human":
                    before,after=state.human_hp,damage(state.human_hp, BASE_DAMAGE)
                    state.human_hp=after
                else:
                    before,after=state.ai_hp,damage(state.ai_hp, BASE_DAMAGE)
                    state.ai_hp=after
                state.log.append(f"🔫 {actor_label(actor)}{part} {('자신' if tgt==actor else '상대')} → 💥 {before}→{after}")
            else:
                state.log.append(f"🔫 {actor_label(actor)}{part} {('자신' if tgt==actor else '상대')} → ✨ 공탄")

            # 원작 규칙: "자기에게 쐈고, 공탄"이면 턴 유지
            keep = (tgt == actor and not hit)

    # 턴 결정(수갑 포함)
    resolve_next_turn(state, actor, keep)
    state.last_action=action
    return state

# ---------------------- 채팅 엔진 ----------------------
def normalize_cmd(s: str) -> str:
    if not s:
        return s
    t = unicodedata.normalize("NFKC", s.strip()).lower()
    t = re.sub(r"\s+", " ", t)

    opp_patterns = [
        r"상대(에게|한테)? ?쏘기", r"딜러(에게|한테)? ?쏘기", r"너한테 쏘기", r"너에게 쏘기",
        r"상대 ?사격", r"딜러 ?사격", r"상대에게 발사", r"딜러에게 발사",
        r"상대 쏴", r"딜러 쏴", r"너한테 쏴", r"너에게 쏴",
    ]
    self_patterns = [
        r"나(에게|한테)? ?쏘기", r"나한테 쏘기", r"자살? ?시도", r"나 ?사격",
        r"나에게 발사", r"나 ?쏴",
    ]
    peek_patterns = [r"돋보기", r"peek", r"탄(을|만)? ?보기", r"다음 ?탄 ?보기"]
    cuffs_patterns = [r"수갑", r"handcuff", r"핸드커프"]
    heal_patterns  = [r"담배", r"smoke", r"힐", r"회복", r"담배 ?피(기|우기)?"]

    def any_match(pats):
        return any(re.fullmatch(p, t) or re.search(rf"^(?:/)?{p}$", t) for p in pats)

    if any_match(opp_patterns): return "상대에게 쏘기"
    if any_match(self_patterns): return "나에게 쏘기"
    if any_match(peek_patterns): return "돋보기"
    if any_match(cuffs_patterns): return "수갑"
    if any_match(heal_patterns):  return "담배"
    return s.strip()

def chat_with_dealer(state_json:str,history:list,user_msg:str):
    state=GameState(**json.loads(state_json))
    user_msg=user_msg.strip()
    cmd = normalize_cmd(user_msg)
    action = ACTION_MAP.get(cmd)

    if action:
        history=history+[[user_msg,""]]
        state=apply_action(state,"human",action)

        # AI 턴 루프
        cnt=0
        while state.turn=="ai" and cnt<10 and not check_end(state):
            a,r=decide_ai_action(state)
            state.log.append(f"🤖 딜러 선택: {a} ({r})")
            state=apply_action(state,"ai",a)
            cnt+=1

        end=check_end(state)
        if end: state.log.append(f"🏁 {end}")

        return history[:-1]+[[user_msg,"\n".join(state.log[-3:])]], json.dumps(asdict(state),ensure_ascii=False), render_log(state)

    else:
        reply=dealer_chat(state,history,user_msg)
        return history+[[user_msg,reply]], json.dumps(asdict(state),ensure_ascii=False), render_log(state)

# ---------------------- 라운드 진행(버튼용) ----------------------
def next_round(state_json:str, history:list):
    state = GameState(**json.loads(state_json))
    # 아직 라운드 중이면 막기
    if state.magazine and not check_end(state):
        state.log.append("⚠️ 아직 라운드 진행 중입니다. 탄을 모두 소모하거나 누군가 쓰러지면 다음 라운드로 이동하세요.")
        return history, json.dumps(asdict(state),ensure_ascii=False), render_log(state)

    verdict = check_end(state)
    if verdict:
        state.log.append(f"🧾 라운드 {state.round} 결과: {verdict}")

    new_state = new_round(prev=state)
    new_state.log.insert(1, "➡️ 새로운 라운드가 시작됩니다. (HP 리셋, 플레이어 선공)")
    history = history + [["", f"라운드 {new_state.round} 시작! (HP 4/4, 선공: 당신)"]]
    return history, json.dumps(asdict(new_state),ensure_ascii=False), render_log(new_state)

# ---------------------- 시작 ----------------------
def start_game():
    s=new_round()
    s.log.append("🤖 딜러: GPT-4o-mini 준비됨")
    hist=[["","게임 시작! 행동을 채팅으로 입력하거나 버튼을 눌러보세요."]]
    return json.dumps(asdict(s),ensure_ascii=False), hist, render_log(s)

# ---------------------- UI ----------------------
with gr.Blocks() as demo:
    gr.Markdown("# 🔫 Buckshot Roulette - Chat Version")
    gr.Markdown(
        """
            **간단 사용법**
            - **게임 시작/리셋** 버튼으로 라운드를 시작합니다. (HP 4로 리셋, 당신 선공)
            - 행동은 **버튼 클릭** 또는 **채팅 입력**으로 합니다.
            - 사격: `상대에게 쏘기`, `나에게 쏘기`  *(자연어도 인식: “너한테 쏘기”, “나한테 쏘기” 등)*
            - 아이템: `돋보기`, `수갑`, `담배`
            - **아이템은 무료 액션**입니다. 사용해도 **당신 턴이 유지**됩니다.
            - 수갑: 상대의 **다음 턴 1회 스킵** 예약
            - 돋보기: 다음 탄 정보 확인(딜러는 비공개)
            - 담배: HP +1 (최대 4)
            - **사격 규칙**: 기본은 사격 후 턴 교대. 단, **자기에게 쐈는데 공탄**이면 **턴 유지**.
            - **다음 라운드** 버튼: 탄을 다 쓰거나 누가 쓰러지면 눌러서 새 라운드 시작.
            - 아래 **🧾 게임 로그**에서 모든 진행 기록을 확인하세요.
        """
    )
    
    with gr.Row():
        start_btn=gr.Button("게임 시작/리셋")
        next_btn=gr.Button("다음 라운드")

    state_store=gr.State("")
    chatbot=gr.Chatbot(height=360)

    with gr.Row():
        btn_self=gr.Button("나에게 쏘기")
        btn_opp=gr.Button("상대에게 쏘기")
        btn_peek=gr.Button("돋보기")
        btn_cuffs=gr.Button("수갑")
        btn_heal=gr.Button("담배")

    chat_in=gr.Textbox(placeholder="메시지 입력...",lines=2)
    send_btn=gr.Button("보내기",variant="primary")

    gr.Markdown("## 🧾 게임 로그")
    log_box = gr.Textbox(label="게임 로그", lines=16, interactive=False)

    # 초기화 / 다음 라운드
    start_btn.click(start_game,inputs=[],outputs=[state_store,chatbot,log_box])
    next_btn.click(next_round, inputs=[state_store,chatbot], outputs=[chatbot,state_store,log_box])

    # 버튼 → 채팅
    for b,txt in [(btn_self,"나에게 쏘기"),(btn_opp,"상대에게 쏘기"),(btn_peek,"돋보기"),(btn_cuffs,"수갑"),(btn_heal,"담배")]:
        b.click(chat_with_dealer,inputs=[state_store,chatbot,gr.State(txt)],outputs=[chatbot,state_store,log_box])

    # 채팅
    send_btn.click(chat_with_dealer,inputs=[state_store,chatbot,chat_in],outputs=[chatbot,state_store,log_box])
    chat_in.submit(chat_with_dealer,inputs=[state_store,chatbot,chat_in],outputs=[chatbot,state_store,log_box])

if __name__=="__main__":
    demo.launch(share=True)