```batch
    @echo off
    echo Starting Fake News Detection API...
    cd /d "%~dp0.."
    call env_api\Scripts\activate.bat
    echo API Environment activated
    uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
    pause
    ```