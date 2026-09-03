# TrustRAG Frontend V3

Frontend-only redesign. Backend/API contract is unchanged.

## Replace
Copy this `frontend` folder over your existing project's `frontend` folder.

## Run
```powershell
cd "C:\Users\zohai\Desktop\TrustRAG_React_Professional\trustrag\frontend"
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
npm install
npm run dev
```

Backend stays on `http://localhost:8000`; frontend defaults to it unless `VITE_API_URL` is set.
