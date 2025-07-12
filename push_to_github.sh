#!/bin/bash

echo "🚀 Finespresso Analytics - GitHub Push Script"
echo "=============================================="
echo ""
echo "This script will push your Streamlit application to GitHub."
echo "You will need to provide your GitHub credentials."
echo ""
echo "Repository: https://github.com/kaljuvee/finespresso-modelling.git"
echo "Branch: main"
echo ""

# Check if we're in the right directory
if [ ! -f "app.py" ]; then
    echo "❌ Error: app.py not found. Please run this script from the finespresso-streamlit directory."
    exit 1
fi

# Check git status
echo "📋 Checking git status..."
git status

echo ""
echo "🔐 Starting push to GitHub..."
echo "When prompted:"
echo "  Username: kaljuvee"
echo "  Password: [your GitHub Personal Access Token]"
echo ""
echo "💡 If you don't have a Personal Access Token:"
echo "   1. Go to GitHub.com → Settings → Developer settings → Personal access tokens"
echo "   2. Generate new token with 'repo' scope"
echo "   3. Use the token as your password"
echo ""

read -p "Press Enter to continue with the push..."

# Push to GitHub
git push -u origin main

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ Successfully pushed to GitHub!"
    echo ""
    echo "🌐 Next steps:"
    echo "1. Go to https://share.streamlit.io"
    echo "2. Sign in with your GitHub account"
    echo "3. Click 'New app'"
    echo "4. Select repository: kaljuvee/finespresso-modelling"
    echo "5. Set main file: app.py"
    echo "6. Click 'Deploy!'"
    echo ""
    echo "📖 For detailed instructions, see GITHUB_DEPLOYMENT_GUIDE.md"
else
    echo ""
    echo "❌ Push failed. Please check your credentials and try again."
    echo "📖 See GITHUB_DEPLOYMENT_GUIDE.md for troubleshooting."
fi

