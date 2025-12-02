# 🌐 웹 애플리케이션 배포 가이드

이 프로젝트를 외부에서 접속 가능한 링크로 만드는 두 가지 방법을 안내합니다.

---

## 🚀 방법 1: ngrok 사용 (가장 빠름, 임시)
**내 컴퓨터가 켜져 있는 동안만** 외부에서 접속할 수 있는 임시 링크를 생성합니다. 시연이나 테스트용으로 적합합니다.

1.  **ngrok 설치**
    *   [ngrok 홈페이지](https://ngrok.com/download)에서 회원가입 후 다운로드합니다.
    *   터미널에서 압축을 풀고 인증 토큰을 등록합니다 (홈페이지 안내 참조).

2.  **포트 연결**
    *   앱이 실행 중인 상태(`start_app.sh` 실행 중)에서, **새로운 터미널**을 열고 아래 명령어를 입력합니다.
    ```bash
    ngrok http 8000
    ```

3.  **링크 공유**
    *   터미널에 표시되는 `Forwarding` 주소 (예: `https://abcd-1234.ngrok-free.app`)를 복사해서 공유하면 됩니다.

---

## ☁️ 방법 2: Hugging Face Spaces (추천, 영구적)
무료로 AI 웹 앱을 호스팅해주는 서비스입니다. **컴퓨터를 꺼도 링크가 유지**되므로 포트폴리오나 과제 제출용으로 가장 좋습니다.

1.  **Space 생성**
    *   [Hugging Face](https://huggingface.co/new-space)에 접속하여 로그인합니다.
    *   **Space Name**: `pcb-defect-detection` (원하는 이름)
    *   **License**: `MIT`
    *   **Sdk**: **Docker** (중요! Docker를 선택해야 합니다)
    *   `Create Space` 클릭.

2.  **파일 업로드**
    *   생성된 Space 페이지에서 `Files` 탭으로 이동합니다.
    *   `Add file` -> `Upload files`를 클릭합니다.
    *   **내 프로젝트의 모든 파일**을 드래그해서 업로드합니다.
        *   `backend/` 폴더 (특히 `best.pt` 포함 필수!)
        *   `frontend/` 폴더
        *   `Dockerfile` (제가 방금 만들어드린 파일)
        *   `requirements.txt` (backend 폴더 안에 있음)
    *   `Commit changes`를 클릭하여 저장합니다.

3.  **배포 확인**
    *   업로드가 완료되면 자동으로 `Building`이 시작됩니다.
    *   몇 분 후 `Running` 상태가 되면 상단의 `App` 탭에서 웹사이트를 확인할 수 있습니다.
    *   해당 주소를 공유하면 누구나 접속 가능합니다!

---

### 💡 주의사항
*   **모델 파일**: `backend/best.pt` 파일이 반드시 포함되어야 합니다. (용량이 크다면 Git LFS를 사용해야 할 수도 있지만, 100MB 이하라면 웹 업로드로 충분합니다.)
*   **OpenCV 의존성**: `Dockerfile`에 이미 OpenCV 실행을 위한 시스템 라이브러리 설치 설정(`libgl1-mesa-glx`)을 포함시켜 두었습니다.
