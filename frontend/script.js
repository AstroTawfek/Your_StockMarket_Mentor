async function checkBackend() {
    try {
        const res = await fetch('http://localhost:8000/health');
        if (!res.ok) throw new Error(`HTTP error! status: ${res.status}`);
        const data = await res.json();
        return data.status === "Backend is running";
    } catch {
        return false;
    }
}

async function sendMessage() {
    const message = document.getElementById('message').value;
    const language = document.getElementById('language').value;
    const responseElem = document.getElementById('response');

    if (!message) {
        responseElem.textContent = 'Please enter a message.';
        return;
    }

    if (!(await checkBackend())) {
        responseElem.textContent = 'Backend not running. Start it with: cd backend && source venv/bin/activate && uvicorn app:app --reload --port 8000';
        return;
    }

    try {
        const res = await fetch('http://localhost:8000/chat', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ message, language }),
        });
        if (!res.ok) throw new Error(`HTTP error! status: ${res.status}`);
        const data = await res.json();
        responseElem.textContent = data.response;
    } catch (error) {
        responseElem.textContent = 'Error: ' + error.message + '. Check http://localhost:8000/models for available Gemini models or verify GEMINI_API_KEY.';
    }
}

async function analyze() {
    const company_name = document.getElementById('company_name').value;
    const pe_ratio = parseFloat(document.getElementById('pe_ratio').value);
    const revenue_growth = parseFloat(document.getElementById('revenue_growth').value);
    const netinc_growth = parseFloat(document.getElementById('netinc_growth').value);
    const gross_margin = parseFloat(document.getElementById('gross_margin').value);
    const roic = parseFloat(document.getElementById('roic').value);
    const profit_margin = parseFloat(document.getElementById('profit_margin').value);

    const ratingElem = document.getElementById('rating');
    const adviceElem = document.getElementById('advice');

    if (!company_name || isNaN(pe_ratio) || isNaN(revenue_growth) || isNaN(netinc_growth) || isNaN(gross_margin) || isNaN(roic) || isNaN(profit_margin)) {
        ratingElem.textContent = 'Please fill all fields with valid numbers (e.g., PE Ratio: 39.42, percentages like 14.13).';
        return;
    }

    if (!(await checkBackend())) {
        ratingElem.textContent = 'Backend not running. Start it with: cd backend && source venv/bin/activate && uvicorn app:app --reload --port 8000';
        return;
    }

    try {
        const res = await fetch('http://localhost:8000/analyze', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                company_name,
                pe_ratio,
                revenue_growth,
                netinc_growth,
                gross_margin,
                roic,
                profit_margin
            }),
        });
        if (!res.ok) throw new Error(`HTTP error! status: ${res.status}`);
        const data = await res.json();
        ratingElem.textContent = 'Rating: ' + data.rating;
        adviceElem.textContent = 'Advice: ' + data.advice;
    } catch (error) {
        ratingElem.textContent = 'Error: ' + error.message + '. Check http://localhost:8000/models for available Gemini models or verify GEMINI_API_KEY.';
    }
}