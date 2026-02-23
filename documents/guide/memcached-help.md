# Memcached Setup Guide (Windows + Docker)

This guide explains how to install and run Memcached on Windows using
Docker Desktop. It assumes you have zero prior experience with Docker or
Memcached.

------------------------------------------------------------------------

# What is Memcached?

Memcached is an in-memory key-value store used to cache data (like
sessions, API results, etc.) to improve performance and reduce database
load.

Your project uses Memcached for session caching.

------------------------------------------------------------------------

# Recommended Setup: Windows + Docker Desktop

This is the safest and easiest way to run Memcached on Windows.

------------------------------------------------------------------------

## Step 1: Install Docker Desktop

1.  Go to: https://www.docker.com/products/docker-desktop/
2.  Download Docker Desktop for Windows.
3.  Run the installer.
4.  During installation:
    -   Enable WSL2 integration if prompted.
5.  Restart your computer if required.
6.  Open Docker Desktop and make sure it shows: Docker is running

You are ready to use Docker.

------------------------------------------------------------------------

## Step 2: Start Memcached Container

Open Command Prompt or PowerShell and run:

docker run -d --name echoai-memcached -p 11211:11211
memcached:1.6.40-alpine memcached -m 256 -p 11211 -v

What this does:

-   -d → runs container in background
-   --name echoai-memcached → gives container a name
-   -p 11211:11211 → exposes port 11211 to your PC
-   -m 256 → allocates 256MB memory to Memcached
-   -v → verbose logging

If successful, Docker will return a long container ID.

------------------------------------------------------------------------

## Step 3: Verify Memcached is Running

Run:

docker ps

You should see:

CONTAINER ID IMAGE PORTS xxxxxxx memcached:1.6.40-alpine
0.0.0.0:11211-\>11211/tcp

If you see this, Memcached is running correctly.

------------------------------------------------------------------------

## Step 4: Add Environment Variables to Your Project

Update your `.env` file:

MEMCACHED_ENABLED=true MEMCACHED_HOSTS=localhost:11211
MEMCACHED_TTL=1800 MEMCACHED_FALLBACK=true MEMCACHED_POOL_SIZE=10
MEMCACHED_TIMEOUT=5

Explanation:

-   MEMCACHED_ENABLED → turns cache on
-   MEMCACHED_HOSTS → where Memcached is running
-   MEMCACHED_TTL → cache time-to-live (seconds)
-   MEMCACHED_FALLBACK → app continues if cache fails
-   MEMCACHED_POOL_SIZE → connection pool size
-   MEMCACHED_TIMEOUT → connection timeout in seconds

Save the file.

------------------------------------------------------------------------

## Step 5: Start Your Application

Run your backend normally (example):

uvicorn apps.gateway.main:app --reload

------------------------------------------------------------------------

## Step 6: Validate Cache Health

Open browser or Postman:

GET http://localhost:8000/health/cache

Expected response:

{ "cache_available": true, "test_set": "ok", "test_get": "ok",
"test_delete": "ok" }

If you see this → Memcached is fully working.

------------------------------------------------------------------------

# Optional: Create Easy Start Script (Windows)

Create a file called: start_memcached.bat

Add:

@echo off docker rm -f echoai-memcached \>nul 2\>nul docker run -d
--name echoai-memcached -p 11211:11211 memcached:1.6.40-alpine memcached
-m 256 -p 11211 -v echo Memcached started on localhost:11211

Now double-click this file anytime to restart Memcached.

------------------------------------------------------------------------

# How to Stop Memcached

docker stop echoai-memcached

------------------------------------------------------------------------

# How to Remove Memcached Completely

docker rm -f echoai-memcached

------------------------------------------------------------------------

# Troubleshooting

If Docker is not running: - Open Docker Desktop manually.

If port 11211 is already in use: - Stop any existing Memcached
instance - Or restart Docker

If /health/cache shows false: - Ensure Docker container is running
(docker ps) - Ensure .env values are correct - Restart backend

------------------------------------------------------------------------

# You're Done

Your Windows machine is now running a production-grade Linux Memcached
instance through Docker.


---
---

# Additional Setup Option: Using WSL (Windows Subsystem for Linux)

This section verifies and expands the WSL-based setup.  
These steps are correct, but a few additional installation steps are required for first-time WSL users.

---

## Step 0: Install WSL (If Not Already Installed)

Open PowerShell as Administrator and run:

wsl --install

Restart your computer if prompted.

After restart:
- Open Ubuntu from Start Menu
- Complete username + password setup

---

## Step 1: Update Packages (Recommended)

Inside Ubuntu terminal:

sudo apt update
sudo apt upgrade -y

---

## Step 2: Install Memcached

sudo apt install -y memcached

---

## Step 3: Install Netcat (Required for Testing)

Some Ubuntu versions don’t include netcat by default:

sudo apt install -y netcat

---

## Step 4: Start Memcached Service

sudo service memcached start

Verify:

sudo service memcached status

You should see:
Active: active (running)

---

## Step 5: Test Memcached Connection

echo "stats" | nc localhost 11211

If working, you’ll see stats output including:

curr_items
get_hits
get_misses
bytes

If you see stats → Memcached is working correctly.

---

## Step 6: Update Your .env File

DATABASE_URL=postgresql+asyncpg://echoai:echoai_dev@localhost:5432/echoai
MEMCACHED_ENABLED=true
MEMCACHED_HOSTS=localhost:11211
AUTH_ENFORCEMENT=optional

Important:

If your backend runs on Windows (not inside WSL), 
replace localhost with the WSL IP address:

Find WSL IP:

ip addr show eth0

Then update:

MEMCACHED_HOSTS=<WSL_IP>:11211

If backend also runs inside WSL → keep localhost.

---

# 3 Ways to Verify Memcached is Working

## 1. Health Endpoint (Easiest)

http://localhost:8000/health/cache

Expected:

{
  "status": "ok",
  "cache_available": true,
  "test_set": "ok",
  "test_get": "ok"
}

---

## 2. Command Line Stats

echo "stats" | nc localhost 11211

Watch these values:

- curr_items     → Number of items in cache
- get_hits       → Cache hits
- get_misses     → Cache misses
- bytes          → Memory used

---

## 3. Live Monitoring

watch -n 2 'echo "stats" | nc localhost 11211 | grep -E "curr_items|get_hits|get_misses|bytes"'

Updates every 2 seconds.

---

# How It Works

User Request → Check Memcached  
            → HIT? Return immediately  
            → MISS? Query PostgreSQL  
                     → Store in Memcached  
                     → Return response  

---

# Where is Data Stored?

Memcached:
- Stored in RAM only
- Lost on restart
- Extremely fast

PostgreSQL:
- Stored on disk
- Permanent storage
- Source of truth

Memcached = Fast temporary cache  
PostgreSQL = Permanent storage

---

# Important Notes

1. Memcached in WSL stops when WSL shuts down.
2. Data is NOT persistent (RAM only).
3. For production environments, use Linux servers or Docker.
4. WSL is suitable for development only.


---
---

# Additional Setup Option: Running Memcached via Docker (Beginner Detailed Guide)

This section explains step-by-step how to start Memcached on localhost:11211 
using the Docker image `memcached:1.6.40-alpine`.

This assumes you have zero prior Docker knowledge.

---

## Step 1: Install Docker Desktop (One-Time Setup)

1. Go to:
   https://www.docker.com/products/docker-desktop/

2. Download Docker Desktop for Windows.

3. Run the installer.

4. During installation:
   - Enable WSL2 integration if prompted.

5. Restart your computer if required.

6. After restart:
   - Open Docker Desktop
   - Wait until it shows:
     Docker is running

If Docker is not running, Memcached will not start.

---

## Step 2: Open Command Prompt

1. Press Win + R
2. Type: cmd
3. Press Enter

A terminal window will open.

---

## Step 3: Run Memcached Container

Copy and paste the following command:

docker run -d --name echoai-memcached -p 11211:11211 memcached:1.6.40-alpine memcached -m 256 -p 11211 -v

Press Enter.

---

### What This Command Does

docker run → Create and start container  
-d → Run in background  
--name echoai-memcached → Name of container  
-p 11211:11211 → Expose port 11211 to your PC  
memcached:1.6.40-alpine → Official Memcached image  
-m 256 → Allocate 256MB RAM  
-v → Enable verbose logs  

---

### First-Time Behavior

If this is your first time running it:

Docker will automatically download the image.
This is normal and may take a minute.

---

## Step 4: Verify Container is Running

Run:

docker ps

You should see something like:

CONTAINER ID   IMAGE                     PORTS  
abc123         memcached:1.6.40-alpine   0.0.0.0:11211->11211/tcp  

If you see this → Memcached is running successfully.

---

## Step 5: View Logs (Optional)

docker logs echoai-memcached

You should see startup logs confirming Memcached started.

---

## Step 6: Test Port 11211

Optional test using Telnet:

Enable Telnet (if not installed):

dism /online /Enable-Feature /FeatureName:TelnetClient

Then run:

telnet localhost 11211

Ensure port 11211 is free ===>  netstat -ano | findstr :11211

If it connects → Memcached is reachable.

To exit Telnet:
Press Ctrl + ]
Then type:
quit

---

## How to Stop Memcached

docker stop echoai-memcached

---

## How to Remove It Completely

docker rm -f echoai-memcached

---

## How to Restart It Later

docker start echoai-memcached

---

## Final Result

Memcached is now running at:

localhost:11211

Your .env should contain:

MEMCACHED_ENABLED=true  
MEMCACHED_HOSTS=localhost:11211  

---
