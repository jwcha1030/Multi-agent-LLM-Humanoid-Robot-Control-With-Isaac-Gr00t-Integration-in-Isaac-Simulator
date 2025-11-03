#!/bin/bash
# Multi-Agent System Installation Script

echo "=========================================="
echo "Multi-Agent System Installation"
echo "=========================================="
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if we're in the right directory
if [ ! -f "multiagent_system.py" ]; then
    echo -e "${RED} Error: multiagent_system.py not found${NC}"
    echo "Please run this script from the isaac-sim-with-groot-tutorial directory"
    exit 1
fi

echo -e "${GREEN} Found multiagent_system.py${NC}"
echo ""

# Check for .env file
if [ -f ".env" ]; then
    echo -e "${GREEN} Found .env file${NC}"
    if grep -q "OPENAI_API_KEY" .env; then
        echo -e "${GREEN} OPENAI_API_KEY found in .env${NC}"
    else
        echo -e "${YELLOW}  OPENAI_API_KEY not found in .env${NC}"
        echo "Please add your OpenAI API key to .env:"
        echo "  echo 'OPENAI_API_KEY=sk-proj-YOUR-KEY' >> .env"
    fi
else
    echo -e "${YELLOW}  .env file not found${NC}"
    echo "Creating .env file..."
    read -p "Enter your OpenAI API key (or press Enter to skip): " api_key
    if [ -n "$api_key" ]; then
        echo "OPENAI_API_KEY=$api_key" > .env
        echo -e "${GREEN} Created .env file${NC}"
    else
        echo -e "${YELLOW}  Skipped. You'll need to create .env manually${NC}"
    fi
fi

echo ""
echo "=========================================="
echo "Installing Python Dependencies"
echo "=========================================="
echo ""

# Check if isaac-py is available
if command -v isaac-py &> /dev/null; then
    echo -e "${GREEN} Found isaac-py${NC}"
    echo "Installing dependencies with isaac-py..."
    isaac-py -m pip install -r requirements_multiagent.txt
    
    if [ $? -eq 0 ]; then
        echo -e "${GREEN} Dependencies installed successfully${NC}"
    else
        echo -e "${RED} Failed to install dependencies${NC}"
        exit 1
    fi
else
    echo -e "${YELLOW}  isaac-py not found${NC}"
    echo "Trying with regular pip..."
    pip install -r requirements_multiagent.txt
    
    if [ $? -eq 0 ]; then
        echo -e "${GREEN} Dependencies installed successfully${NC}"
    else
        echo -e "${RED} Failed to install dependencies${NC}"
        exit 1
    fi
fi

echo ""
echo "=========================================="
echo "Verifying Installation"
echo "=========================================="
echo ""

# Test imports
echo "Testing Python imports..."
python3 << 'EOF'
import sys
try:
    from crewai import Agent, Task, Crew
    print(" CrewAI imported successfully")
except ImportError as e:
    print(f" Failed to import CrewAI: {e}")
    sys.exit(1)

try:
    from langchain_openai import ChatOpenAI
    print(" LangChain OpenAI imported successfully")
except ImportError as e:
    print(f" Failed to import LangChain OpenAI: {e}")
    sys.exit(1)

try:
    from dotenv import load_dotenv
    print(" python-dotenv imported successfully")
except ImportError as e:
    print(f" Failed to import dotenv: {e}")
    sys.exit(1)

try:
    import openai
    print(" OpenAI imported successfully")
except ImportError as e:
    print(f" Failed to import OpenAI: {e}")
    sys.exit(1)

print("\n All imports successful!")
EOF

if [ $? -ne 0 ]; then
    echo -e "${RED} Import verification failed${NC}"
    exit 1
fi

echo ""
echo "=========================================="
echo "Checking Action Library"
echo "=========================================="
echo ""

if [ -d "action_library" ]; then
    action_count=$(ls -1 action_library/*.json 2>/dev/null | wc -l)
    echo -e "${GREEN} Action library found: $action_count actions${NC}"
    
    if [ $action_count -gt 0 ]; then
        echo "Available actions:"
        ls -1 action_library/*.json | sed 's/.*\///' | sed 's/\.json$//' | sed 's/^/  - /'
    fi
else
    echo -e "${YELLOW}  Action library directory not found${NC}"
    echo "Creating action_library directory..."
    mkdir -p action_library
    echo -e "${GREEN} Created action_library directory${NC}"
    echo "Record actions using: isaac-py run_simulation_with_recorder.py"
fi

echo ""
echo "=========================================="
echo "Installation Complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo ""
echo "1. Test without simulator:"
echo "   ${GREEN}python test_multiagent.py${NC}"
echo ""
echo "2. Run with Isaac Sim:"
echo "   ${GREEN}isaac-py run_multiagent_sim.py${NC}"
echo ""
echo "3. Read documentation:"
echo "   - MULTIAGENT_SETUP.md     (Detailed guide)"
echo "   - MULTIAGENT_QUICK_REF.md (Quick reference)"
echo ""
echo "Need to record actions?"
echo "   ${GREEN}isaac-py run_simulation_with_recorder.py${NC}"
echo ""
echo -e "${GREEN} Setup complete! Happy coding!${NC}"
echo ""
