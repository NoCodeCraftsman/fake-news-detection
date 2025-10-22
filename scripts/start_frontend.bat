```batch
    @echo off
    echo Starting Streamlit Frontend...
    cd /d "%~dp0.."
    call env_frontend\Scripts\activate.bat
    echo Frontend Environment activated
    streamlit run frontend/app.py
    pause
    ```