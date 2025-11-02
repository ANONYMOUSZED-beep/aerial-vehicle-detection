@echo off
echo 🚀 Setting up Git repository for GitHub...
echo ==================================================

REM Initialize Git repository
echo 📂 Initializing Git repository...
git init

REM Add all files
echo ➕ Adding files to Git...
git add .

REM Create initial commit
echo 💾 Creating initial commit...
git commit -m "Initial commit: RF-DETR Traffic Monitoring System - Features: RF-DETR vehicle detection engine, Interactive web dashboard with Dash/Plotly, Real-time traffic monitoring and analytics, Video processing with progress tracking, Zone-based traffic analysis, Live camera integration, Comprehensive documentation"

echo ✅ Git repository setup complete!
echo.
echo 🔗 Next steps to push to GitHub:
echo 1. Create a new repository on GitHub
echo 2. Copy the repository URL
echo 3. Run these commands:
echo.
echo    git remote add origin ^<your-github-repo-url^>
echo    git branch -M main
echo    git push -u origin main
echo.
echo 📋 Example:
echo    git remote add origin https://github.com/yourusername/rf-detr-traffic-monitoring.git
echo    git branch -M main
echo    git push -u origin main
echo.
echo 🎯 Repository is ready for GitHub!
pause