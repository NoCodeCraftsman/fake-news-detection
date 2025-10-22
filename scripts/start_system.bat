```batch
    @echo off
    echo Starting Complete Fake News Detection System...
    echo.
    echo Starting API in background...
    start "" cmd /c "call scripts\start_api.bat"
    
    timeout /t 5 /nobreak > nul
    echo.
    echo Starting Frontend...
    call scripts\start_frontend.bat
    ```