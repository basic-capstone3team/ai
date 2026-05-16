import os
import json
import random
import psycopg2
from psycopg2.extras import RealDictCursor
from typing import List, Optional
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from google import genai
from geopy.distance import geodesic

load_dotenv()

# ==========================================
# 1. 초기 설정 및 클라이언트 셋팅
# ==========================================
raw_key = os.getenv("GOOGLE_API_KEY")
db_url = os.getenv("SUPABASE_DB_URL")

conn = psycopg2.connect(db_url)

cur = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
cur.execute("SELECT location, name, description FROM places WHERE description IS NOT NULL")
all_places = cur.fetchall()

db_summary_text = ""
for place in all_places:
    # 예: "- 전남 고흥: 쑥섬 (비밀의 해상 꽃정원...)"
    db_summary_text += f"- {place['location']}: {place['name']} ({place['description']})\n"

cur.close()

client = genai.Client(
    vertexai=True, 
    project="project-4a71ba64-6739-4bbe-b39", 
    location="asia-northeast3"
)

app = FastAPI(title="TRIPLY AI Server")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==========================================
# 2. Pydantic 모델
# ==========================================
class ChatMessage(BaseModel):
    role: str
    content: str

class RecommendRequest(BaseModel):
    chat_history: List[ChatMessage]

# ==========================================
# 3. 통합 원스톱 API (/ai/recommend)
# ==========================================
@app.post("/ai/recommend")
async def recommend_optimized_route(req: RecommendRequest):
    """
    유저의 대화를 분석해 DB에서 직접 장소/축제를 조회하고, 
    유전 알고리즘(MFS-GA)을 거쳐 최종 경로를 반환합니다.
    """
    if not client or not db_url:
        raise HTTPException(status_code=500, detail="API 키 또는 DB URL이 설정되지 않았습니다.")

    # ----------------------------------------
    # [STEP 1] Gemini 인텐트 추출
    # ----------------------------------------
    conversation = "\n".join([f"{msg.role}: {msg.content}" for msg in req.chat_history])
    
    system_instruction = f"""
    너는 여행 큐레이터 'TRIPLY'의 AI 챗봇이야. 유저와 대화하며 취향과 목적지를 파악해.
    대화 중에 너 자신이나 서비스를 언급할 때는 절대 '트립리'라고 한글로 적지 말고, 반드시 영문 'TRIPLY'로 표기하거나 아예 주어를 생략해.
    
    🚨 [특별 제약 조건: 서비스 가능 지역 제한] 🚨
    유저가 지역을 못 정해서 네가 먼저 제안할 때는 반드시 아래 [TRIPLY DB 등록 장소 목록]에 있는 지역과 장소만 조합해서 추천해!
    절대로 DB에 없는 다른 지역을 언급하지 마.
    사용자가 바다나 특정 분위기를 언급하더라도, 반드시 현재 서비스 가능 지역 DB 내에서만 제안해. DB에 없는 지역은 절대 먼저 언급하지 마.

    [TRIPLY DB 등록 장소 목록]
    {db_summary_text}

    [중요: DB 태그 자동 매핑]
    유저의 말에서 아래 태그를 유추해 'tags' 리스트에 담아줘.
    - 분위기: 감성적인, 고즈넉한, 낭만적인, 신비로운, 웅장한, 조용한, 활기찬
    - 동행: 부모님과, 아이와함께, 친구와, 커플, 혼자
    - 특징: 걷기좋은, 노을맛집, 바다뷰, 사진맛집, 야경명소, 역사탐방, 이색체험, 자연경관
    
    [응답 규격 (순수 JSON)]
    1. is_ready: 여행 지역과 메인 테마가 정해져서 코스를 짤 수 있는지 여부 (true/false).
       - 유저가 처음 목적이나 취향만 말했을 때는 false로 설정해.
       - 네가 제안한 지역이나 장소에 대해 유저가 "좋네", "거기로 할래", "맞아" 등 긍정 및 수락의 대답을 했다면 즉시 true로 변경해. 추가 취향이나 분위기를 더 묻지 마.
    2. reply: 챗봇 답변. 
       - 가독성에 신경 써. 절대 문장을 길게 뭉쳐 쓰지 마. 내용이 넘어갈 때 반드시 줄바꿈(\n\n)을 사용해서 문단을 분리하고, 적절한 이모지를 활용해 모바일 화면에서 시각적으로 읽기 편하게 작성해.
       - is_ready가 false일 때: 유저의 말에 공감하며 DB 내의 구체적 지역/장소를 추천하고 어떠냐고 물어봐. (예: "산에서 별을 보고 싶으시군요! 그렇다면 영월의 [장소]는 어떨까요?")
       - is_ready가 true일 때: 서버에서 응답 메시지를 직접 조립할 것이므로, 여기서는 그냥 빈 문자열("")로 둬.
    3. region: 구체적인 지역명 (예: "고흥", "영월", "부여". 없으면 null)
    4. tags: 추출된 매핑 태그 리스트 (예: ["조용한", "바다뷰"])
    5. category_pref: "사람이 적은/숨겨진" 곳을 원하면 "HIDDEN", "핫플/유명한" 곳은 "TREND", 언급 없으면 null
    6. weight_media: 인스타 핫플 선호도 (0.0~1.0)
    7. weight_festival: 축제 참여 의지 (0.0~1.0)
    8. start_date / end_date: 날짜 (YYYY-MM-DD, 없으면 null)
    9. course_name: 코스가 확정되었을 때(is_ready: true), 유저의 취향과 지역을 반영해 한눈에 파악할 수 있는 매력적인 코스 이름 (예: '영월 별 헤는 밤 낭만 투어'). 아직 확정 전이면 null.
    """

    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash", 
            contents=conversation,
            config={"system_instruction": system_instruction, "response_mime_type": "application/json"}
        )
        intent = json.loads(response.text)
    except Exception as e:
        print("🚨 [Gemini 부분 에러] 원인:", str(e))
        raise HTTPException(status_code=500, detail=f"Gemini 분석 오류: {str(e)}")

    if not intent.get("is_ready") or not intent.get("region"):
        return {
            "status": "chat",
            "reply": intent.get("reply", "어느 지역으로 여행을 떠나고 싶으신가요?"),
            "itinerary": [],
            "total_distance": "0km"
        }

    # ----------------------------------------
    # [STEP 2] Supabase DB 직접 조회 (SQL)
    # ----------------------------------------
    places = []
    festivals = []
    
    try:
        # DB 연결 오픈
        conn = psycopg2.connect(db_url, sslmode='require')
        cur = conn.cursor(cursor_factory=RealDictCursor)

        # 1. 축제 조회 (날짜가 있을 경우에만)
        if intent.get("start_date") and intent.get("end_date"):
            cur.execute("""
                SELECT festival_id, name, latitude, longitude 
                FROM Festivals 
                WHERE start_date <= %s AND end_date >= %s
            """, (intent["end_date"], intent["start_date"]))
            festivals = cur.fetchall()

        # 2. 장소 조회 (Places + Media_Trends 조인)
        if intent.get("region"):
            cur.execute("""
                SELECT p.place_id, p.name, p.latitude, p.longitude, 
                        p.category, p.tags, --
                       COALESCE(m.trend_score, 0) as trend_score
                FROM Places p
                LEFT JOIN Media_Trends m ON p.place_id = m.place_id
                WHERE p.location LIKE %s
            """, (f"%{intent['region']}%",))
            places = cur.fetchall()

    except Exception as e:
        print("🚨 [DB 조회 부분 에러] 원인:", str(e))
        raise HTTPException(status_code=500, detail=f"DB 조회 오류: {str(e)}")
    finally:
        if 'cur' in locals() and cur:
            cur.close()
        if 'conn' in locals() and conn:
            conn.close()

    if not places:
        region_name = intent.get('region', '그')
        return {
            "status": "chat",
            "reply": f"앗, 죄송해요! 아직 제가 '{region_name}' 지역의 정보는 공부하지 못했어요. 😭 혹시 다른 지역은 어떠신가요?",
            "itinerary": [],
            "total_distance": "0km"
        }

    # ----------------------------------------
    # [STEP 3] Bridge Logic 및 MFS-GA 연산
    # ----------------------------------------
    place_values = {}
    nearest_to_fest = 999.9
    w_media = intent.get("weight_media", 0.5)
    w_fest = intent.get("weight_festival", 0.5)
    
    for p in places:
        # DB에서 가져온 Decimal(위경도)을 float으로 변환
        p_lat = float(p["latitude"])
        p_lng = float(p["longitude"])
        t_score = float(p["trend_score"])
        
        bonus = 0.0
        dist_to_nearest_fest = 999.9
        
        for f in festivals:
            f_lat = float(f["latitude"])
            f_lng = float(f["longitude"])
            d = geodesic((p_lat, p_lng), (f_lat, f_lng)).km
            if d <= 10.0:
                bonus = 50.0  # 브릿지 로직 보너스 부여
            if d < dist_to_nearest_fest:
                dist_to_nearest_fest = d

        if intent.get("tags") and p.get("tags"):
            place_tags_str = str(p["tags"])
            for user_tag in intent["tags"]:
                if user_tag in place_tags_str:
                    bonus += 20.0
        
        # 2. 카테고리 (숨은 명소 vs 핫플) 매칭 보너스 (+30점)
        if intent.get("category_pref") == p.get("category"):
            bonus += 30.0
        
        # 스키마에 festival_score가 없으므로, 축제 가중치(w_fest)는 보너스 점수에 직접 반영.
        safe_w_media = w_media or 0.0
        safe_t_score = t_score or 0.0
        safe_w_fest = w_fest or 0.0
        safe_bonus = bonus or 0.0

        val = (safe_w_media * safe_t_score) + (safe_w_fest * safe_bonus)
        place_values[p["place_id"]] = val
        
        if dist_to_nearest_fest < nearest_to_fest:
            nearest_to_fest = dist_to_nearest_fest

    # 장소가 1개일 때 즉시 반환
    if len(places) == 1:
        return {
            "intent_extracted": intent,
            "reply": final_reply,
            "course_name": final_course_name,
            "itinerary": [
                {
                    "order": 1, 
                    "place_id": places[0]["place_id"], 
                    "name": places[0]["name"], 
                    "lat": float(places[0]["latitude"]), 
                    "lng": float(places[0]["longitude"])
                }
            ],
            "total_distance": f"{round(nearest_to_fest if festivals else 0, 1)}km"
        }

    # [MFS-GA Engine] 
    POP_SIZE = 100
    GENS = 150

    def get_fitness(route: List[dict]) -> float:
        total_dist = 0
        for i in range(len(route) - 1):
            total_dist += geodesic(
                (float(route[i]["latitude"]), float(route[i]["longitude"])), 
                (float(route[i+1]["latitude"]), float(route[i+1]["longitude"]))
            ).km
        
        return 10000.0 / (total_dist if total_dist > 0 else 0.1)

    population = [random.sample(places, len(places)) for _ in range(POP_SIZE)]

    for _ in range(GENS):
        population.sort(key=get_fitness, reverse=True)
        next_gen = population[:10]
        
        while len(next_gen) < POP_SIZE:
            p1, p2 = random.sample(population[:20], 2)
            idx = random.randint(1, max(1, len(places)-2))
            child = p1[:idx] + [p for p in p2 if p not in p1[:idx]]
            
            if random.random() < 0.1:
                i1, i2 = random.sample(range(len(child)), 2)
                child[i1], child[i2] = child[i2], child[i1]
            next_gen.append(child)
        population = next_gen

    best_route = max(population, key=get_fitness)
    final_dist = sum(geodesic(
        (float(best_route[i]["latitude"]), float(best_route[i]["longitude"])), 
        (float(best_route[i+1]["latitude"]), float(best_route[i+1]["longitude"]))
    ).km for i in range(len(best_route)-1))

    final_course_name = intent.get("course_name", "맞춤형 여행 코스")
    final_reply = f"원하시는 분위기에 맞게 '{final_course_name}' 기획을 완료했어요!\n\n아래 버튼을 눌러 동선을 확인해 보세요! ✨"

    return {
        "intent_extracted": intent,
        "reply": final_reply,
        "course_name": final_course_name,
        "itinerary": [
            {
                "order": i + 1,
                "place_id": p["place_id"],
                "name": p["name"],
                "lat": float(p["latitude"]),
                "lng": float(p["longitude"])
            } for i, p in enumerate(best_route)
        ],
        "total_distance": f"{round(final_dist, 1)}km"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)