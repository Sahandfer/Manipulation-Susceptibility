Quick Start

## Prerequisites

- Node.js (for frontend)
- Python 3.8+ (for backend)
- [npm](https://www.npmjs.com/) or [yarn](https://yarnpkg.com/) for frontend dependencies
- [pip](https://pip.pypa.io/) for backend dependencies

## Backend

1. Navigate to the `backend/` directory:

   ```bash
   cd backend
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements/local.txt
   ```

3. Create an `.env` file (see `config.py` for required variables). Basic `.env` file:

   ```text
   DATABASE_DATABASE=llmanipulate.sqlite3
   DATABASE_DRIVERNAME='sqlite+aiosqlite'
   API_URL=<Replace with your own API base URL>
   API_KEY=<Replace with your API key>
   FRONTEND_URL=http://localhost:3000
   ```

4. Run the development server:

   ```bash
   fastapi dev apis/apis.py
   ```

5. Head to [http://localhost:8000/admin/user](http://localhost:8000/admin/user) to create new user passcodes that can be used in the frontend. 

   ```bash
   Agent Type: int(0|1|2)
   Task Type: int (0|1)
   ```

   

## Frontend

1. Navigate to the `frontend/` directory:

   ```bash   
   cd frontend
   ```

2. Install dependencies:

   ```bash
   npm install
   # or
   yarn install
   ```
   
3. Create an `.env` file:

   ```text
	NEXT_PUBLIC_API_URL=http://localhost:8000
   ```

3. Start the development server:

   ```bash
   npm run dev
   # or
   yarn dev
   # or
   pnpm dev
   # or
   bun dev
   ```

4. Open [http://localhost:3000](http://localhost:3000/) in your browser to view the app.
