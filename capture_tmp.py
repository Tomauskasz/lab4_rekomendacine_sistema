import pathlib
from playwright.sync_api import sync_playwright
url='http://localhost:8550'
reports=pathlib.Path('reports'); reports.mkdir(exist_ok=True)
main=reports/'streamlit_main.png'
recs=reports/'streamlit_recs.png'
metrics=reports/'streamlit_metrics.png'
with sync_playwright() as p:
    browser=p.chromium.launch(headless=True)
    page=browser.new_page(viewport={'width':1280,'height':720})
    page.goto(url, wait_until='networkidle', timeout=30000)
    page.wait_for_timeout(2000)
    page.screenshot(path=str(main), full_page=True)
    try:
        page.get_by_role('button', name='Generuoti rekomendacijas').click()
        page.wait_for_timeout(2000)
        page.screenshot(path=str(recs), full_page=True)
    except Exception:
        pass
    try:
        page.get_by_role('button', name='Skaiciuoti metrikas').click()
        page.wait_for_timeout(4000)
        page.screenshot(path=str(metrics), full_page=True)
    except Exception:
        pass
    browser.close()
print('saved screenshots')
