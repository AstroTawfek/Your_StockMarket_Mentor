async function sendMessage() {
    const message = document.getElementById('message').value;
    const language = document.getElementById('language').value;
    const responseElem = document.getElementById('response');

    if (!message) {
        responseElem.textContent = 'Please enter a message.';
        return;
    }

    try {
        const res = await fetch('http://localhost:8000/chat', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ message, language }),
        });
        if (!res.ok) {
            throw new Error(`HTTP error! status: ${res.status}`);
        }
        const data = await res.json();
        responseElem.textContent = data.response;
    } catch (error) {
        responseElem.textContent = 'Network error: ' + error.message + '. Ensure backend is running on http://localhost:8000.';
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
        ratingElem.textContent = 'Please fill all fields with valid numbers.';
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
        if (!res.ok) {
            throw new Error(`HTTP error! status: ${res.status}`);
        }
        const data = await res.json();
        ratingElem.textContent = 'Rating: ' + data.rating;
        adviceElem.textContent = 'Advice: ' + data.advice;
    } catch (error) {
        ratingElem.textContent = 'Network error: ' + error.message + '. Ensure backend is running on http://localhost:8000.';
    }
}