<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <!-- <title>Stock Market Mentor</title> -->
</head>
<body>
    <h1>Stock Market Mentor</h1>
    <p>Welcome to Stock Market Mentor, a web application designed to empower users with stock market insights through an interactive chatbot and advanced stock analysis. This repository contains the code and resources for a comprehensive platform aimed at empowering users to make informed investment decisions.</p>
    
    <h2>Table of Contents</h2>
    <ul>
        <li><a href="#introduction">Introduction</a></li>
        <li><a href="#features">Features</a></li>
        <li><a href="#technologies-used">Technologies Used</a></li>
        <li><a href="#usage">Usage</a></li>
        <li><a href="#contributing">Contributing</a></li>
        <li><a href="#team-members">Team Members</a></li>
        <li><a href="#license">License</a></li>
        <li><a href="#acknowledgements">Acknowledgements</a></li>
    </ul>

    <h2 id="introduction">Introduction</h2>
    <p>Stock Market Mentor is a user-friendly platform that combines machine learning and natural language processing to deliver stock market analysis and financial advice. It uses an XGBoost model trained on real stock market data to predict analyst ratings (Buy/Sell/Hold) and integrates the Google Gemini API (gemini-1.5-flash) for tailored responses in English or Bengali (Banglish). The app features a modern, animated interface with a professional stock-market-inspired design, making it engaging for both novice and experienced investors.</p>

    <h2 id="features">Features</h2>
    <ul>
        <li>Interactive Chatbot: Ask unlimited financial questions in English or Banglish, receiving clear and insightful responses.</li>
        <li>Stock Analysis: Input key company metrics (e.g., PE Ratio, Revenue Growth) to obtain accurate analyst ratings and technical advice.</li>
        <li>Modern UI: Responsive design with side-by-side chat and analysis sections, consistent input fields, and dynamic animations (bounce-in title, slide-up sections, glowing buttons).</li>
        <li>Reliable Backend: FastAPI server with robust error handling ensures seamless predictions and API interactions.</li>
    </ul>

    <h2 id="technologies-used">Technologies Used</h2>
    <ul>
        <li><strong>Frontend:</strong> HTML, CSS, JavaScript</li>
        <li><strong>Backend:</strong> Python, FastAPI, XGBoost, Google Gemini API (gemini-1.5-flash)</li>
        <li><strong>Data Processing:</strong> Pandas, NumPy, Scikit-learn</li>
        <li><strong>Version Control:</strong> Git, GitHub</li>
    </ul>

    <h2 id="usage">Usage</h2>
    <p>
        <ul>
            <li><strong>Chat:</strong> Select a language (English or Banglish) from the dropdown, type a financial question (e.g., "What are stock dividends?") in the input box, and click Send to receive a response. Supports unlimited queries within Google Gemini free tier limits (~60 requests/minute).</li>
            <li><strong>Analysis:</strong> Enter a company name (e.g., "Apple Inc.") and financial metrics (e.g., PE Ratio: 39.42, Revenue Growth: 14.13%) in the provided fields, then click Analyze to view the predicted analyst rating (Buy/Sell/Hold) and technical advice.</li>
            <li><strong>UI Experience:</strong> Enjoy a sleek interface with smooth animations, including a bouncing title, sliding sections, and interactive button effects, optimized for a professional stock market context.</li>
        </ul>
    </p>

    <h2 id="contributing">Contributing</h2>
    <p>Contributions are welcome! To contribute:</p>
    <ol>
        <li>Fork the repository.</li>
        <li>Create a feature branch (<code>git checkout -b feature/YourFeature</code>).</li>
        <li>Commit your changes (<code>git commit -m "Add YourFeature"</code>).</li>
        <li>Push to the branch (<code>git push origin feature/YourFeature</code>).</li>
        <li>Open a pull request.</li>
    </ol>
    <p>Please ensure your code follows the project’s style and includes relevant tests.</p>

    <h2 id="team-members">Team Members</h2>
    <ul>
        <li><strong>Mohammad Ullah Tawfek : </strong> <a href="https://github.com/AstroTawfek">GitHub Profile</a></li>
    </ul>

    <h2 id="license">License</h2>
    <p>This project is licensed under the MIT License. See the <a href="LICENSE">LICENSE</a> file for details.</p>

    <h2 id="acknowledgements">Acknowledgements</h2>
    <ul>
        <li>Google Gemini API for providing natural language processing capabilities.</li>
        <li>XGBoost for enabling accurate stock market predictions.</li>
        <li>The open-source community for tools like FastAPI, Pandas, and Scikit-learn.</li>
    </ul>
</body>
</html>