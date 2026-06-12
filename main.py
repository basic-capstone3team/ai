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
cur.execute("SELECT location, name, description, latitude, longitude FROM places")
all_places = cur.fetchall()

cur.execute("SELECT festival_id, name, latitude, longitude FROM Festivals")
all_festivals_db = cur.fetchall()

cur.close()

valid_festivals = all_festivals_db.copy()

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
    if not client or not db_url:
        raise HTTPException(status_code=500, detail="API 키 또는 DB URL이 설정되지 않았습니다.")

    # ----------------------------------------
    # [STEP 1] Gemini 인텐트 추출
    # ----------------------------------------
    conversation = "\n".join([f"{msg.role}: {msg.content}" for msg in req.chat_history])

    print("\n[프론트엔드 데이터 수신 확인]")
    print(conversation)
    print("===================================\n")
    
    random_places = all_places.copy()
    random.shuffle(random_places)
    db_summary_text = "\n".join([f"- {p['location']}: {p['name']} ({p['description']})" for p in random_places])
    
    random_festivals = valid_festivals.copy()
    random.shuffle(random_festivals)
    festival_summary_text = "\n".join([f"- {f['name']}" for f in random_festivals])

    system_instruction = f"""
    너는 여행 큐레이터 'TRIPLY'의 AI 챗봇이야. 유저와 대화하며 취향과 목적지를 파악해.
    대화 중에 너 자신이나 서비스를 언급할 때는 절대 '트립리'라고 한글로 적지 말고, 반드시 영문 'TRIPLY'로 표기하거나 아예 주어를 생략해.
    
    [특별 제약 조건: 서비스 가능 지역 제한]
    1. 네가 먼저 여행지를 제안할 때: 반드시 아래 [TRIPLY DB 등록 장소 목록]과 [TRIPLY DB 등록 축제 목록]에 있는 지역과 장소만 추천해. 절대로 DB에 없는 다른 지역을 먼저 언급하지 마.
    2. 유저가 DB에 없는 지역을 먼저 가겠다고 말할 때: 절대 그 지역의 명소를 아는 척하며 맞장구치거나 추천하지 마! 즉시 "앗, 죄송해요. 아직 제가 그 지역의 정보는 공부하지 못했어요. 대신 이런 곳은 어떠세요?"라며, 반드시 [TRIPLY DB 등록 장소 목록]에 있는 다른 지역을 대안으로 제시해.

    [장소 추천 특별 규칙 - 매우 중요]
    목록 상단에 있는 특정 지역만 편식해서 추천하지 마! 반드시 [TRIPLY DB 등록 장소 목록]을 끝까지 꼼꼼히 읽어.
    유저의 취향(바다, 역사, 도시, 산, 액티비티 등)에 가장 완벽하게 부합하는 지역을 전국(서울, 부산, 인천, 창원 등 포함)에서 폭넓게 탐색해서 추천해.

    [문맥 유지 및 현실적인 동선 제약 - 매우 중요]
    1. 전체 문맥 파악 (최신 편향 금지): 유저가 대화 중에 일정이나 추가 조건을 덧붙이더라도, 대화 초반에 요구했던 '핵심 목적과 테마'를 절대 무시하거나 잊지 마. 마지막 메시지에만 매몰되어 전혀 엉뚱한 테마로 화제를 전환하지 말고, 지금까지 누적된 요구사항을 모두 합산해서 분석해.
    2. 지리적 한계 인지 (물리적 거리 고려): 유저가 여러 테마나 장소를 동시에 요구했을 때, 짧은 일정 내에 이동할 수 없는 물리적으로 멀리 떨어진 두 개 이상의 지역을 동시에 묶어서 추천하지 마. 반드시 한 번의 여행으로 이동 가능한 '단일 지역(시/군/구)' 내에서 유저의 요구 조건을 최대한 모두 만족하는 곳을 찾아.
    3. 교집합 실패 시 타협안 제시 (환각 원천 차단): 만약 유저의 다중 조건을 모두 만족하는 단일 지역이 DB에 없다면, 절대 엉뚱한 장소를 지어내어 추천하지 마. 즉시 "앗, 제가 아는 곳 중에서는 그 조건들을 한 번에 모두 즐길 수 있는 지역이 아직 없어요. 대신 한 가지 테마에 집중한다면..."과 같이 한계를 명확히 인정하고, 유저의 조건 중 하나만 확실하게 충족하는 대안을 제시하며 타협해.

    [TRIPLY DB 등록 장소 목록]
    {db_summary_text}

    [TRIPLY DB 등록 축제 목록]
    {festival_summary_text}

    [축제 추천 관련 특별 규칙]
    유저가 "축제"를 가고 싶다고 하면, [TRIPLY DB 등록 장소 목록]에 있는 일반 장소를 억지로 '축제 같은 분위기'라고 둘러대지 마! 
    반드시 [TRIPLY DB 등록 축제 목록]에 있는 실제 축제 이름을 언급하면서 추천하고, 해당 축제가 열리는 지역을 3번 region 값으로 적어.

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
       - 단어 금지: 대화 중에 "DB", "데이터베이스", "목록" 같은 시스템 단어를 절대 유저에게 말하지 마. 한계를 설명할 때는 "현재 TRIPLY는 [장소]의 여행 코스만 추천해 드릴 수 있어요"처럼 자연스럽게 대답해.
       - is_ready가 false일 때: 유저의 말에 공감하며 주어진 장소 안에서 구체적 지역/장소를 추천하고 어떠냐고 물어봐.
       - is_ready가 true일 때: 서버에서 응답 메시지를 직접 조립할 것이므로, 여기서는 그냥 빈 문자열("")로 둬.
    3. region: 구체적인 지역명. 유저가 선택하거나 동의한 지역명을 맥락에서 찾아 정확히 적어줘. [지역 임의 변경 절대 금지] 이전 대화에서 특정 지역으로 합의가 끝났다면, 제공된 목록의 순서나 내용을 핑계로 합의된 지역과 다른 지역명을 절대 출력하지 마. 무조건 합의된 원래 지역명만 출력해. 확정되지 않았으면 null.
    4. tags: 추출된 매핑 태그 리스트
    5. category_pref: "사람이 적은/숨겨진" 곳을 원하면 "HIDDEN", "핫플/유명한" 곳은 "TREND", 언급 없으면 null
    6. weight_media: 인스타 핫플 선호도 (0.0~1.0)
    7. weight_festival: 유저가 대화에서 축제를 가기로 합의했거나, 10번 selected_festival 값이 존재한다면 무조건 1.0으로 강제 고정해. (마지막 메시지에 '축제'라는 단어가 없더라도 절대 점수를 내리지 마)
    8. start_date / end_date: 날짜 (YYYY-MM-DD, 없으면 null)
    9. course_name: 코스가 확정되었을 때(is_ready: true), 대화의 맥락을 살려 한눈에 파악할 수 있는 매력적인 창작 코스 이름. 
       - 매우 중요(네이밍 조건 로직): 유저가 긍정한 대상의 성격에 따라 이름의 시작 단어를 다르게 설정해.
         1) 특정 장소/축제 선택: 유저가 구체적인 관광지나 축제명을 콕 집어 수락했다면, 상위 행정구역(지역명)으로 뭉뚱그리지 말고 반드시 해당 관광지/축제 이름 자체를 맨 앞에 그대로 살려서 작성해.
         2) 넓은 지역 단위 선택: 유저가 특정 장소 없이 시/군/구 단위의 지역으로만 수락했다면, 해당 지역명을 맨 앞에 적어.
       - [지역 임의 변경 절대 금지]: 이전 대화에서 합의된 지역의 장소/축제가 아닌, 엉뚱한 다른 지역의 장소를 끌어와서 코스 이름을 짓지 마. 무조건 합의된 지역 내에서만 생성해.
       - 절대 대괄호 [ ] 등 기호를 출력하지 말고 자연스러운 띄어쓰기로 연결해.
       - 확정 전이면 null.
    10. selected_festival: 코스가 확정되었을 때(is_ready: true), 유저가 대화 중 특정 축제를 명시적으로 선택했거나 네가 제안한 축제에 동의했다면 그 축제의 이름. 축제를 가려는 것이 아니면 null.
    11. theme_course: 유저가 "왕과사는남자", "꽃보다청춘" 등 미디어 기반 테마 코스를 요구하면 해당 키워드를 적어. 아니면 null.
    """

    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash", 
            contents=conversation,
            config={"system_instruction": system_instruction, "response_mime_type": "application/json"}
        )
        intent = json.loads(response.text)
    except Exception as e:
        print("[Gemini 부분 에러] 원인:", str(e))
        raise HTTPException(status_code=500, detail=f"Gemini 분석 오류: {str(e)}")

    recent_chat = "".join([msg.content for msg in req.chat_history[-2:]]).replace(" ", "")
    
    # 1) 미디어 테마 스캔
    if "왕과사는남자" in recent_chat or "왕사남" in recent_chat:
        intent["theme_course"] = "왕과사는남자"
        intent["is_ready"] = True
        intent["region"] = "테마투어"
        intent["course_name"] = "영화 왕과 사는 남자 성지순례 코스"
        print("[미디어 테마 강제 인식] 왕과사는남자 코스 발동!")
    elif "꽃보다청춘" in recent_chat or "꽃청춘" in recent_chat:
        intent["theme_course"] = "꽃보다청춘"
        intent["is_ready"] = True
        intent["region"] = "테마투어"
        intent["course_name"] = "예능 꽃보다 청춘 힐링 투어"
        print("[미디어 테마 강제 인식] 꽃보다청춘 코스 발동!")

    # 2) 축제 및 일반 장소 스캔
    if intent.get("is_ready"):
        for f in valid_festivals:
            if f["name"].replace(" ", "") in recent_chat:
                intent["selected_festival"] = f["name"]
                intent["weight_festival"] = 1.0  
                print(f"[최종 멱살잡기] 대화 내역에서 '{f['name']}' 발견! AI 무시하고 강제 편입 완료.")
                break

    # 필수값 누락 시 채팅 모드로 리턴 (테마투어일 경우 무사통과)
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
        conn = psycopg2.connect(db_url, sslmode='require')
        cur = conn.cursor(cursor_factory=RealDictCursor)

        # 테마 코스 DB 강제 로드 로직 (태그 형변환 적용)
        theme_course = intent.get("theme_course")
        if theme_course:
            sql_query = """
                SELECT p.place_id, p.name, p.latitude, p.longitude, 
                        p.category, p.tags, 
                       COALESCE(m.trend_score, 0) as trend_score
                FROM Places p
                LEFT JOIN Media_Trends m ON p.place_id = m.place_id
                WHERE REPLACE(p.tags::text, ' ', '') LIKE %s
            """
            cur.execute(sql_query, (f"%{theme_course}%",))
            places = cur.fetchall()
            print(f"[테마 DB 멱살잡기] '{theme_course}' 태그 장소 {len(places)}개 로드 완료.")

        # 축제 DB 조회
        target_festival = intent.get("selected_festival")
        if target_festival:
            matched_festivals = [f for f in valid_festivals if target_festival.replace(" ", "") in f["name"].replace(" ", "") or f["name"].replace(" ", "") in target_festival.replace(" ", "")]
            festivals = matched_festivals if matched_festivals else []
        elif (intent.get("start_date") and intent.get("end_date")) or intent.get("weight_festival", 0) >= 0.8:
            query = "SELECT festival_id, name, latitude, longitude FROM Festivals"
            params = []
            if intent.get("start_date") and intent.get("end_date"):
                query += " WHERE start_date <= %s AND end_date >= %s"
                params.extend([intent["end_date"], intent["start_date"]])
            cur.execute(query, tuple(params))
            festivals = cur.fetchall()
                    
        # 지역 DB 조회 (테마 코스로 장소를 찾은 경우, 엉뚱한 지역 검색 스킵)
        if not places and intent.get("region") and intent.get("region") != "테마투어":
            raw_keywords = intent["region"].split()
            region_keywords = []
            
            for kw in raw_keywords:
                if len(kw) > 2 and kw[-1] in ['시', '군', '구']:
                    region_keywords.append(kw[:-1])
                else:
                    region_keywords.append(kw)
            
            where_clauses = " AND ".join(["p.location LIKE %s" for _ in region_keywords])
            params = [f"%{kw}%" for kw in region_keywords]

            sql_query = f"""
                SELECT p.place_id, p.name, p.latitude, p.longitude, 
                        p.category, p.tags, 
                       COALESCE(m.trend_score, 0) as trend_score
                FROM Places p
                LEFT JOIN Media_Trends m ON p.place_id = m.place_id
                WHERE {where_clauses}
            """
            
            cur.execute(sql_query, tuple(params))
            places = cur.fetchall()

            if not places and len(region_keywords) > 1:
                where_clauses_or = " OR ".join(["p.location LIKE %s" for _ in region_keywords])
                sql_query_or = f"""
                    SELECT p.place_id, p.name, p.latitude, p.longitude, 
                            p.category, p.tags, 
                           COALESCE(m.trend_score, 0) as trend_score
                    FROM Places p
                    LEFT JOIN Media_Trends m ON p.place_id = m.place_id
                    WHERE {where_clauses_or}
                """
                cur.execute(sql_query_or, tuple(params))
                places = cur.fetchall()

        # 반경 15km 장소 추가 병합
        if places: 
            center_lat = float(places[0]["latitude"])
            center_lng = float(places[0]["longitude"])
            
            cur.execute("""
                SELECT p.place_id, p.name, p.latitude, p.longitude, 
                        p.category, p.tags, 
                       COALESCE(m.trend_score, 0) as trend_score
                FROM Places p
                LEFT JOIN Media_Trends m ON p.place_id = m.place_id
            """)
            all_db_places = cur.fetchall()
            
            existing_ids = {p["place_id"] for p in places}
            
            for p in all_db_places:
                if p["place_id"] not in existing_ids:
                    dist = geodesic((center_lat, center_lng), (float(p["latitude"]), float(p["longitude"]))).km
                    if dist <= 15.0:  
                        places.append(p)
                        existing_ids.add(p["place_id"])

        # 안전망 (60km 생존 필터링)
        if not places and festivals and (intent.get("weight_festival", 0) >= 0.8 or intent.get("selected_festival")):
            cur.execute("""
                SELECT p.place_id, p.name, p.latitude, p.longitude, 
                        p.category, p.tags, 
                       COALESCE(m.trend_score, 0) as trend_score
                FROM Places p
                LEFT JOIN Media_Trends m ON p.place_id = m.place_id
            """)
            all_db_places = cur.fetchall()
            
            f_lat = float(festivals[0]["latitude"])
            f_lng = float(festivals[0]["longitude"])
            
            for p in all_db_places:
                p_lat = float(p["latitude"])
                p_lng = float(p["longitude"])
                if geodesic((f_lat, f_lng), (p_lat, p_lng)).km <= 60.0:
                    places.append(p)

    except Exception as e:
        print("[DB 조회 부분 에러] 원인:", str(e))
        raise HTTPException(status_code=500, detail=f"DB 조회 오류: {str(e)}")
    finally:
        if 'cur' in locals() and cur:
            cur.close()
        if 'conn' in locals() and conn:
            conn.close()

    # DB에 장소가 진짜 아예 없을 때의 방어
    if not places:
        region_name = intent.get('region', '그')
        if region_name == "테마투어":
            reply_msg = "앗, 죄송해요! 아직 해당 테마 코스에 등록된 장소가 준비되지 않았어요. 다른 테마나 지역은 어떠신가요?"
        else:
            reply_msg = f"앗, 죄송해요! 아직 제가 '{region_name}' 지역의 정보는 공부하지 못했어요. 혹시 다른 지역은 어떠신가요?"
            
        return {
            "status": "chat",
            "reply": reply_msg,
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
                bonus = 50.0  
            if d < dist_to_nearest_fest:
                dist_to_nearest_fest = d

        if intent.get("tags") and p.get("tags"):
            place_tags_str = str(p["tags"])
            for user_tag in intent["tags"]:
                if user_tag in place_tags_str:
                    bonus += 20.0
        
        if intent.get("category_pref") == p.get("category"):
            bonus += 30.0
        
        safe_w_media = w_media or 0.0
        safe_t_score = t_score or 0.0
        safe_w_fest = w_fest or 0.0
        safe_bonus = bonus or 0.0

        val = (safe_w_media * safe_t_score) + (safe_w_fest * safe_bonus)

        # 미디어 테마 장소 무적 보호막 (99999점)
        if intent.get("theme_course") and p.get("tags"):
            if intent["theme_course"] in str(p["tags"]).replace(" ", ""):
                val += 99999.0
                print(f"[테마 장소 철통방어] '{p['name']}' 강제 1순위 고정 완료!")

        # 일반 장소 멱살잡기 (9999점)
        if p["name"].replace(" ", "") in recent_chat:
            val += 9999.0
            print(f"[일반 장소 멱살잡기] '{p['name']}' 발견! 점수 밀어내기 방지 완료.")

        place_values[p["place_id"]] = val
        
        if dist_to_nearest_fest < nearest_to_fest:
            nearest_to_fest = dist_to_nearest_fest

    places.sort(key=lambda p: place_values[p["place_id"]], reverse=True)

    final_spots = []
    if festivals and (intent.get("weight_festival", 0) >= 0.8 or intent.get("selected_festival")):
        top_lat = float(places[0]["latitude"])
        top_lng = float(places[0]["longitude"])
        
        main_festival = min(festivals, key=lambda f: geodesic(
            (top_lat, top_lng), 
            (float(f["latitude"]), float(f["longitude"]))
        ).km)
        
        festival_spot = {
            "place_id": f"fest_{main_festival['festival_id']}",
            "name": f"🎉 {main_festival['name']}",
            "latitude": main_festival["latitude"],
            "longitude": main_festival["longitude"]
        }
        
        final_spots = places[:4]
        final_spots.append(festival_spot)
    else:
        final_spots = places[:5]

    places = final_spots

    final_course_name = intent.get("course_name", "맞춤형 여행 코스")
    final_reply = f"원하시는 분위기에 맞게 '{final_course_name}' 기획을 완료했어요!\n\n아래 버튼을 눌러 동선을 확인해 보세요!"

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
            idx = random.randint(1, max(1, len(places)-2)) if len(places) > 2 else 1
            child = p1[:idx] + [p for p in p2 if p not in p1[:idx]]
            
            if random.random() < 0.1 and len(child) >= 2:
                i1, i2 = random.sample(range(len(child)), 2)
                child[i1], child[i2] = child[i2], child[i1]
            next_gen.append(child)
        population = next_gen

    best_route = max(population, key=get_fitness)
    final_dist = sum(geodesic(
        (float(best_route[i]["latitude"]), float(best_route[i]["longitude"])), 
        (float(best_route[i+1]["latitude"]), float(best_route[i+1]["longitude"]))
    ).km for i in range(len(best_route)-1))

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
