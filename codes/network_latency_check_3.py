import asyncio
import csv
import time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from playwright.async_api import async_playwright

# ==========================================
# 설정 구역
# ==========================================
CONCURRENT_LIMIT = 150  
semaphore = asyncio.Semaphore(CONCURRENT_LIMIT)

FIVE_G_USERS = 50
FOUR_G_USERS = 50
THREE_G_USERS = 50
TEST_DURATION = 60

URL = "https://eqyvtgycwxeirvve7cldswlk6a0eungg.lambda-url.ap-northeast-2.on.aws/"
timestamp = time.strftime("%Y%m%d_%H%M%S")
SUMMARY_FILE = f"test_summary_report_{timestamp}.csv"
RAW_FILE = f"test_raw_data_{timestamp}.csv"
PLOT_FILE = f"latency_chart_{timestamp}.png"

all_latency_results = []
USER_CONFIG = {"5G": FIVE_G_USERS, "4G": FOUR_G_USERS, "3G": THREE_G_USERS}

async def monitor_network(user_id, network_type, playwright):
    async with semaphore:
        browser = await playwright.chromium.launch(headless=True)
        # 독립적인 컨텍스트 생성 (캐시 분리)
        context = await browser.new_context()
        
        await context.route("**/*", lambda route: route.continue_(
            headers={**route.request.headers, "Cache-Control": "no-cache", "Pragma": "no-cache"}
        ))
        
        page = await context.new_page()

        # [네트워크 설정 테이블]
        speeds = {
            "3G": {
                "down": int(1.6 * 1000 * 1000 / 8 * 0.9), 
                "up": int(0.75 * 1000 * 1000 / 8 * 0.9), 
                "lat": int(150 * 3.75)
            },
            "4G": {"down": int(15*1024*1024//8), "up": int(7.5*1024*1024//8), "lat": 40},
            "5G": {"down": int(450*1024*1024//8), "up": int(50*1024*1024//8), "lat": 15}
        }
        
        s = speeds[network_type]

        if network_type == "3G":
            await page.add_init_script(f"""
                Object.defineProperty(navigator, 'connection', {{
                    get: () => ({{ effectiveType: '3g', downlink: 1.44, rtt: {s['lat']} }})
                }});
            """)

        client = await page.context.new_cdp_session(page)
        await client.send("Network.emulateNetworkConditions", {
            "offline": False, "downloadThroughput": s["down"], 
            "uploadThroughput": s["up"], "latency": s["lat"]
        })

        request_map = {}
        page.on("request", lambda req: request_map.update({req: time.perf_counter()}))
        
        async def on_response(res):
            req = res.request
            # GET 메서드이면서 대상 API 경로인 경우에만 지연 시간 측정
            if "/get-button-status" in req.url and req.method == "GET" and req in request_map:
                latency = int((time.perf_counter() - request_map.pop(req)) * 1000)
                all_latency_results.append({"User_ID": user_id, "Network_Type": network_type, "Latency_ms": latency})

        page.on("response", on_response)
        await page.goto(URL)
        await asyncio.sleep(TEST_DURATION)
        await browser.close()
        print(f"유저 {user_id} ({network_type}) 테스트 완료.")

async def main():
    async with async_playwright() as p:
        tasks = []
        uid = 1
        for _ in range(FIVE_G_USERS): tasks.append(monitor_network(uid, "5G", p)); uid+=1
        for _ in range(FOUR_G_USERS): tasks.append(monitor_network(uid, "4G", p)); uid+=1
        for _ in range(THREE_G_USERS): tasks.append(monitor_network(uid, "3G", p)); uid+=1
        
        print(f" 테스트 시작 (총 {len(tasks)}명, 동시 실행 {CONCURRENT_LIMIT}명 제한)")
        await asyncio.gather(*tasks)

        # 데이터 저장
        with open(RAW_FILE, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["User_ID", "Network_Type", "Latency_ms"])
            writer.writeheader(); writer.writerows(all_latency_results)

        with open(SUMMARY_FILE, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["Network_Type", "Config_Users", "Actual_Requests", "Min_ms", "Median_ms", "Max_ms", "Mean_ms"])
            means = []
            for net in ["5G", "4G", "3G"]:
                # 데이터 정제: 0ms 노이즈 제거
                data = [r["Latency_ms"] for r in all_latency_results if r["Network_Type"] == net and r["Latency_ms"] > 0]
                if data:
                    writer.writerow([net, USER_CONFIG[net], len(data), np.min(data), int(np.median(data)), np.max(data), round(np.mean(data), 2)])
                    means.append(np.mean(data))
                else:
                    means.append(0)

        plt.figure(figsize=(8, 6))
        plt.bar(["5G", "4G", "3G"], means, color=['skyblue', 'orange', 'lightgreen'])
        plt.title('Average API Latency by Network Type (Cache Ignored)')
        plt.ylabel('Latency (ms)')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.savefig(PLOT_FILE)
        
        print(f"\n 테스트 완료! 데이터 파일: {RAW_FILE}, {SUMMARY_FILE}")
        print(f" 그래프 이미지: {PLOT_FILE}")

if __name__ == "__main__":
    asyncio.run(main())