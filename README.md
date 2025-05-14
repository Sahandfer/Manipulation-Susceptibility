# Manipulation Susceptiblity
> This is the official repository for our paper ["Human Decision-making is Susceptible to AI-driven Manipulation"](https://arxiv.org/abs/2502.07663)

<img src="https://img.shields.io/badge/Journal-TBD-278ea5" alt="venue"/> <img src="https://img.shields.io/badge/Status-Under Review-success" alt="status"/> <img src="https://img.shields.io/badge/Contributions-Welcome-red"> <img src="https://img.shields.io/badge/Last%20Updated-2025--05--12-2D333B" alt="update"/>

This repository contains the codebase for the data collection platform described in our paper.

The project is organized into two main components:

- `frontend/`: Next.js-based web application
- `backend/`: FastAPI-based backend server

## Quick Start

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

## Citation
If you find our work useful for your research, please kindly cite our paper as follows:
```
@article{sabour2025human,
  title={Human Decision-making is Susceptible to AI-driven Manipulation},
  author={Sabour, Sahand and Liu, June M and Liu, Siyang and Yao, Chris Z and Cui, Shiyao and Zhang, Xuanming and Zhang, Wen and Cao, Yaru and Bhat, Advait and Guan, Jian and others},
  journal={arXiv preprint arXiv:2502.07663},
  year={2025}
}
```
