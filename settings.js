const Setting = {
    GEMINI: {
        API_KEY: process.env.GEMINI_API_KEY || "----------------",
        MODEL_NAME: process.env.GEMINI_MODEL || "gemini-3.6-flash",
    },

    ANTHROPIC: {
        API_KEY: process.env.ANTHROPIC_API_KEY || "----------------",
        MODEL_NAME: process.env.ANTHROPIC_MODEL || "claude-sonnet-4-6",
    },

    OLLAMA: {
        MODEL_NAME: process.env.OLLAMA_MODEL || "llama3.1:8b",
        BASE_URL: process.env.OLLAMA_BASE_URL || "http://localhost:11434",
    },

    MCP: {
        SERVER: process.env.MCP_SERVER_URL || "http://localhost:3000/mcp",
    },

    MODEL: {
        PROVIDER: process.env.MODEL_PROVIDER || "GEMINI",
    }

};

export default Setting;