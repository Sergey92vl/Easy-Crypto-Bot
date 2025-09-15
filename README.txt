Instructions for use:
1. Download EasyCryptoBot.py file
2. Install the dependencies:
   
bash
pip install ccxt pandas numpy python-dotenv requests schedule matplotlib

4. Create a .env file for API keys:

env
BINANCE_API_KEY=your_api_key_here
BINANCE_API_SECRET=your_api_secret_here

4. Launch the agent

bash
# English, paper trade
python crypto_trading_agent.py --language en --mode paper

# Russian language, paper trade
python crypto_trading_agent.py --language ru --mode paper

# Real trading (requires API keys)
python crypto_trading_agent.py --language en --mode live


Features:
✅ Multilingual support (English, Russian)

✅ Two trading strategies (RSI, MACD)

✅ Paper and real trading

✅ Risk management

✅ Logging of operations

✅ Support for multiple exchanges via CCXT

✅ Multithreaded architecture

✅ Easy to use in a single file
