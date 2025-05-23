// File: api/generate-text.js

export default async function handler(req, res) {
    // Only allow POST requests
    if (req.method !== 'POST') {
        res.setHeader('Allow', ['POST']);
        return res.status(405).end(`Method ${req.method} Not Allowed`);
    }

    const { promptText } = req.body;

    if (!promptText) {
        return res.status(400).json({ error: "Missing 'promptText' in request body" });
    }

    // Retrieve the API key from Vercel Environment Variables
    // Ensure you have set VERCEL_GEMINI_API_KEY in your Vercel project settings.
    const apiKey = process.env.VERCEL_GEMINI_API_KEY;

    if (!apiKey) {
        console.error("Gemini API Key is not configured in Vercel environment variables (VERCEL_GEMINI_API_KEY).");
        return res.status(500).json({ error: "API key not configured on the server." });
    }

    const geminiApiUrl = `https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key=${apiKey}`;
    
    const chatHistory = [{ role: "user", parts: [{ text: promptText }] }];
    const payload = { contents: chatHistory };

    try {
        const geminiResponse = await fetch(geminiApiUrl, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        });

        const responseData = await geminiResponse.json();

        if (!geminiResponse.ok) {
            // Forward the error from Gemini API if possible
            console.error("Error from Gemini API:", responseData);
            const errorMessage = responseData.error?.message || `Gemini API request failed with status ${geminiResponse.status}`;
            return res.status(geminiResponse.status).json({ error: errorMessage, details: responseData });
        }
        
        // Forward the successful response from Gemini API
        return res.status(200).json(responseData);

    } catch (error) {
        console.error("Error calling Gemini API from serverless function:", error);
        return res.status(500).json({ error: "Failed to call Gemini API.", details: error.message });
    }
}
