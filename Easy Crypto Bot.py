#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import argparse
import json
import time
import threading
import logging
from typing import Dict, List, Optional
from datetime import datetime
import pandas as pd
import numpy as np
import ccxt
from dotenv import load_dotenv

# ===== КОНФИГУРАЦИЯ И НАСТРОЙКИ =====

class Config:
    """Класс для управления конфигурацией"""
    def __init__(self):
        self.settings = {
            'default_language': 'en',
            'trading_mode': 'paper',
            'default_exchange': 'binance',
            'watchlist': ['BTC/USDT', 'ETH/USDT', 'BNB/USDT'],
            'update_interval': 5,
            'analysis_interval': 10,
            'monitoring_interval': 60
        }
    
    def get_setting(self, key):
        return self.settings.get(key)

# ===== МНОГОЯЗЫЧНАЯ ПОДДЕРЖКА =====

class LanguageManager:
    def __init__(self, language: str = 'en'):
        self.language = language
        self.translations = self._load_translations()
    
    def _load_translations(self) -> Dict:
        translations = {
            'en': {
                'welcome': '🚀 Welcome to AI Crypto Trading Agent',
                'initializing_agent': 'Initializing trading agent...',
                'shutdown': 'Shutting down agent...',
                'error': '❌ Error',
                'initializing_components': 'Initializing components...',
                'api_keys_required': 'API keys required for live trading',
                'api_keys_missing': 'API keys are missing',
                'initialization_complete': '✅ Initialization complete',
                'starting_agent': 'Starting trading agent',
                'data_error': 'Data error',
                'trading_error': 'Trading error',
                'monitoring_error': 'Monitoring error',
                'shutting_down': 'Shutting down...',
                'shutdown_complete': '✅ Shutdown complete',
                'buy_signal': '📈 Buy signal detected',
                'sell_signal': '📉 Sell signal detected',
                'stop_loss_triggered': '🔴 Stop loss triggered',
                'take_profit_triggered': '🟢 Take profit triggered',
                'connection_success': '✅ Connected to exchange successfully',
                'connection_failed': '❌ Connection to exchange failed'
            },
            'ru': {
                'welcome': '🚀 Добро пожаловать в AI агент для торговли криптовалютой',
                'initializing_agent': 'Инициализация торгового агента...',
                'shutdown': 'Завершение работы агента...',
                'error': '❌ Ошибка',
                'initializing_components': 'Инициализация компонентов...',
                'api_keys_required': 'API ключи необходимы для реальной торговли',
                'api_keys_missing': 'API ключи отсутствуют',
                'initialization_complete': '✅ Инициализация завершена',
                'starting_agent': 'Запуск торгового агента',
                'data_error': 'Ошибка данных',
                'trading_error': 'Ошибка торговли',
                'monitoring_error': 'Ошибка мониторинга',
                'shutting_down': 'Завершение работы...',
                'shutdown_complete': '✅ Завершение работы завершено',
                'buy_signal': '📈 Обнаружен сигнал на покупку',
                'sell_signal': '📉 Обнаружен сигнал на продажу',
                'stop_loss_triggered': '🔴 Сработал стоп-лосс',
                'take_profit_triggered': '🟢 Сработал тейк-профит',
                'connection_success': '✅ Успешное подключение к бирже',
                'connection_failed': '❌ Ошибка подключения к бирже'
            }
        }
        return translations.get(self.language, translations['en'])
    
    def get_text(self, key: str) -> str:
        return self.translations.get(key, key)
    
    def set_language(self, language: str):
        self.language = language
        self.translations = self._load_translations()

# ===== ЛОГГИРОВАНИЕ =====

def setup_logger(name: str, language: str = 'en') -> logging.Logger:
    """Настройка логгера с поддержкой языка"""
    logger = logging.getLogger(name)
    
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        
        # Форматирование для консоли
        console_handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
        # Файловый логгер
        file_handler = logging.FileHandler(f'trading_agent_{language}.log')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger

# ===== БАЗОВЫЕ СТРАТЕГИИ =====

class BaseStrategy:
    """Базовый класс для торговых стратегий"""
    def __init__(self):
        self.name = "base_strategy"
    
    def generate_signal(self, data: pd.DataFrame) -> str:
        """Генерация торгового сигнала"""
        return 'hold'

class RSIStrategy(BaseStrategy):
    """RSI стратегия"""
    def __init__(self, period: int = 14, overbought: int = 70, oversold: int = 30):
        super().__init__()
        self.period = period
        self.overbought = overbought
        self.oversold = oversold
        self.name = "rsi_strategy"
    
    def calculate_rsi(self, data: pd.Series) -> pd.Series:
        """Расчет RSI индикатора"""
        delta = data.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=self.period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=self.period).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def generate_signal(self, data: pd.DataFrame) -> str:
        """Генерация торгового сигнала на основе RSI"""
        if len(data) < self.period + 1:
            return 'hold'
        
        try:
            rsi = self.calculate_rsi(data['close'])
            current_rsi = rsi.iloc[-1]
            
            if pd.isna(current_rsi):
                return 'hold'
                
            if current_rsi < self.oversold:
                return 'buy'
            elif current_rsi > self.overbought:
                return 'sell'
            else:
                return 'hold'
        except Exception as e:
            return 'hold'

class MACDStrategy(BaseStrategy):
    """MACD стратегия"""
    def __init__(self, fast_period: int = 12, slow_period: int = 26, signal_period: int = 9):
        super().__init__()
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.signal_period = signal_period
        self.name = "macd_strategy"
    
    def calculate_macd(self, data: pd.Series) -> tuple:
        """Расчет MACD индикатора"""
        ema_fast = data.ewm(span=self.fast_period).mean()
        ema_slow = data.ewm(span=self.slow_period).mean()
        macd = ema_fast - ema_slow
        signal = macd.ewm(span=self.signal_period).mean()
        histogram = macd - signal
        return macd, signal, histogram
    
    def generate_signal(self, data: pd.DataFrame) -> str:
        """Генерация торгового сигнала на основе MACD"""
        if len(data) < self.slow_period + self.signal_period:
            return 'hold'
        
        try:
            macd, signal, _ = self.calculate_macd(data['close'])
            
            if macd.iloc[-1] > signal.iloc[-1] and macd.iloc[-2] <= signal.iloc[-2]:
                return 'buy'
            elif macd.iloc[-1] < signal.iloc[-1] and macd.iloc[-2] >= signal.iloc[-2]:
                return 'sell'
            else:
                return 'hold'
        except Exception as e:
            return 'hold'

# ===== ОБРАБОТКА ДАННЫХ =====

class DataHandler:
    """Класс для обработки рыночных данных"""
    def __init__(self, exchange: str, language: str = 'en'):
        self.exchange_name = exchange
        self.language = language
        self.lang_manager = LanguageManager(language)
        self.logger = setup_logger('data_handler', language)
        
        self.market_data = {}
        self.watchlist = ['BTC/USDT', 'ETH/USDT', 'BNB/USDT']
        
        # Инициализация exchange
        try:
            exchange_class = getattr(ccxt, exchange)
            self.exchange = exchange_class({
                'enableRateLimit': True,
                'timeout': 30000
            })
            self.logger.info(self.lang_manager.get_text('connection_success'))
        except Exception as e:
            self.logger.error(f"{self.lang_manager.get_text('connection_failed')}: {e}")
            self.exchange = None
    
    def initialize(self, api_key: str = None, api_secret: str = None):
        """Инициализация с API ключами"""
        if api_key and api_secret and self.exchange:
            self.exchange.apiKey = api_key
            self.exchange.secret = api_secret
    
    def update_market_data(self):
        """Обновление рыночных данных"""
        if not self.exchange:
            return
        
        for symbol in self.watchlist:
            try:
                # Получение OHLCV данных
                ohlcv = self.exchange.fetch_ohlcv(symbol, '1m', limit=100)
                df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                df.set_index('timestamp', inplace=True)
                
                self.market_data[symbol] = df
                
            except Exception as e:
                self.logger.error(f"Error updating data for {symbol}: {e}")
    
    def get_market_data(self, symbol: str) -> Optional[pd.DataFrame]:
        """Получение данных по символу"""
        return self.market_data.get(symbol)
    
    def get_watchlist(self) -> List[str]:
        """Получение списка отслеживаемых символов"""
        return self.watchlist
    
    def shutdown(self):
        """Завершение работы"""
        if hasattr(self, 'exchange'):
            try:
                self.exchange.close()
            except:
                pass

# ===== УПРАВЛЕНИЕ РИСКАМИ =====

class RiskManager:
    """Класс для управления рисками"""
    def __init__(self, language: str = 'en'):
        self.language = language
        self.lang_manager = LanguageManager(language)
        self.logger = setup_logger('risk_manager', language)
        
        self.settings = {
            'max_position_size': 0.1,  # 10% от баланса
            'stop_loss': 0.02,  # 2% стоп-лосс
            'take_profit': 0.05,  # 5% тейк-профит
            'max_daily_loss': 0.03,  # 3% максимальный дневной убыток
        }
    
    def check_risk(self, signal: Dict) -> bool:
        """Проверка рисков перед выполнением сделки"""
        # Здесь можно добавить сложную логику проверки рисков
        return True
    
    def calculate_position_size(self, signal: Dict) -> float:
        """Расчет размера позиции"""
        # Упрощенный расчет размера позиции
        return 0.01  # 1% от баланса
    
    def monitor_risk(self):
        """Мониторинг рисков"""
        # Реализация мониторинга рисков
        pass

# ===== ТОРГОВЫЙ ДВИЖОК =====

class TradingEngine:
    """Основной торговый движок"""
    def __init__(self, exchange: str, mode: str, language: str, 
                 data_handler, risk_manager):
        self.exchange_name = exchange
        self.mode = mode
        self.language = language
        self.data_handler = data_handler
        self.risk_manager = risk_manager
        self.lang_manager = LanguageManager(language)
        self.logger = setup_logger('trading_engine', language)
        
        self.strategies: Dict[str, BaseStrategy] = {}
        self.positions: Dict = {}
        self.initialize_strategies()
    
    def initialize_strategies(self):
        """Инициализация торговых стратегий"""
        self.strategies['rsi'] = RSIStrategy()
        self.strategies['macd'] = MACDStrategy()
    
    def initialize(self):
        """Инициализация торгового движка"""
        self.logger.info(self.lang_manager.get_text('initializing_components'))
        
        if self.mode == 'live':
            # Инициализация реального exchange
            try:
                exchange_class = getattr(ccxt, self.exchange_name)
                self.exchange = exchange_class({
                    'apiKey': os.getenv(f'{self.exchange_name.upper()}_API_KEY'),
                    'secret': os.getenv(f'{self.exchange_name.upper()}_API_SECRET'),
                    'enableRateLimit': True
                })
                self.logger.info(f"Live trading initialized for {self.exchange_name}")
            except Exception as e:
                self.logger.error(f"Failed to initialize live trading: {e}")
                self.exchange = None
        else:
            # Бумажная торговля
            self.exchange = None
            self.logger.info("Paper trading mode initialized")
    
    def analyze_market(self) -> List[Dict]:
        """Анализ рынка и генерация торговых сигналов"""
        signals = []
        
        for symbol in self.data_handler.get_watchlist():
            market_data = self.data_handler.get_market_data(symbol)
            
            if market_data is None or len(market_data) < 100:
                continue
            
            # Применение всех стратегий
            for strategy_name, strategy in self.strategies.items():
                signal = strategy.generate_signal(market_data)
                
                if signal != 'hold':
                    signals.append({
                        'symbol': symbol,
                        'strategy': strategy_name,
                        'signal': signal,
                        'price': market_data['close'].iloc[-1],
                        'timestamp': pd.Timestamp.now()
                    })
                    self.logger.info(f"{self.lang_manager.get_text(signal + '_signal')}: {symbol} ({strategy_name})")
        
        return signals
    
    def execute_trades(self, signals: List[Dict]):
        """Выполнение торговых операций"""
        for signal in signals:
            try:
                if self.risk_manager.check_risk(signal):
                    if signal['signal'] == 'buy':
                        self._execute_buy(signal)
                    elif signal['signal'] == 'sell':
                        self._execute_sell(signal)
            except Exception as e:
                self.logger.error(f"Trade execution error: {e}")
    
    def _execute_buy(self, signal: Dict):
        """Выполнение покупки"""
        if self.mode == 'paper':
            self.logger.info(f"📝 PAPER BUY: {signal['symbol']} at {signal['price']:.2f}")
        else:
            # Реальная покупка
            try:
                amount = self.risk_manager.calculate_position_size(signal)
                order = self.exchange.create_market_buy_order(
                    symbol=signal['symbol'],
                    amount=amount
                )
                self.logger.info(f"✅ LIVE BUY: {signal['symbol']} - Amount: {amount}")
            except Exception as e:
                self.logger.error(f"❌ Buy execution failed: {e}")
    
    def _execute_sell(self, signal: Dict):
        """Выполнение продажи"""
        if self.mode == 'paper':
            self.logger.info(f"📝 PAPER SELL: {signal['symbol']} at {signal['price']:.2f}")
        else:
            # Реальная продажа
            try:
                amount = self.risk_manager.calculate_position_size(signal)
                order = self.exchange.create_market_sell_order(
                    symbol=signal['symbol'],
                    amount=amount
                )
                self.logger.info(f"✅ LIVE SELL: {signal['symbol']} - Amount: {amount}")
            except Exception as e:
                self.logger.error(f"❌ Sell execution failed: {e}")
    
    def monitor_positions(self):
        """Мониторинг открытых позиций"""
        # Базовая реализация мониторинга
        pass
    
    def shutdown(self):
        """Завершение работы"""
        if hasattr(self, 'exchange'):
            try:
                self.exchange.close()
            except:
                pass

# ===== ОСНОВНОЙ АГЕНТ =====

class CryptoTradingAgent:
    """Главный класс торгового агента"""
    def __init__(self, language: str = 'en', mode: str = 'paper', exchange: str = 'binance'):
        self.language = language
        self.mode = mode
        self.exchange = exchange
        self.lang_manager = LanguageManager(language)
        self.logger = setup_logger('trading_agent', language)
        
        self.is_running = False
        self.threads = []
        
        # Инициализация компонентов
        self.data_handler = DataHandler(exchange, language)
        self.risk_manager = RiskManager(language)
        self.trading_engine = TradingEngine(
            exchange=exchange,
            mode=mode,
            language=language,
            data_handler=self.data_handler,
            risk_manager=self.risk_manager
        )
    
    def initialize(self):
        """Инициализация агента"""
        self.logger.info(self.lang_manager.get_text('initializing_components'))
        
        # Инициализация API ключей
        api_key = os.getenv(f'{self.exchange.upper()}_API_KEY')
        api_secret = os.getenv(f'{self.exchange.upper()}_API_SECRET')
        
        if self.mode == 'live' and (not api_key or not api_secret):
            self.logger.error(self.lang_manager.get_text('api_keys_required'))
            raise ValueError(self.lang_manager.get_text('api_keys_missing'))
        
        # Инициализация компонентов
        self.data_handler.initialize(api_key, api_secret)
        self.trading_engine.initialize()
        
        self.logger.info(self.lang_manager.get_text('initialization_complete'))
    
    def run(self):
        """Запуск основного цикла торговли"""
        self.is_running = True
        self.logger.info(self.lang_manager.get_text('starting_agent'))
        
        # Запуск потоков
        data_thread = threading.Thread(target=self._data_loop)
        trading_thread = threading.Thread(target=self._trading_loop)
        monitoring_thread = threading.Thread(target=self._monitoring_loop)
        
        self.threads.extend([data_thread, trading_thread, monitoring_thread])
        
        for thread in self.threads:
            thread.daemon = True
            thread.start()
        
        # Основной цикл
        try:
            while self.is_running:
                time.sleep(1)
        except KeyboardInterrupt:
            self.shutdown()
    
    def _data_loop(self):
        """Цикл обработки данных"""
        while self.is_running:
            try:
                self.data_handler.update_market_data()
                time.sleep(5)  # Обновление каждые 5 секунд
            except Exception as e:
                self.logger.error(f"{self.lang_manager.get_text('data_error')}: {e}")
                time.sleep(10)
    
    def _trading_loop(self):
        """Цикл торговли"""
        while self.is_running:
            try:
                signals = self.trading_engine.analyze_market()
                if signals:
                    self.trading_engine.execute_trades(signals)
                time.sleep(10)  # Анализ каждые 10 секунд
            except Exception as e:
                self.logger.error(f"{self.lang_manager.get_text('trading_error')}: {e}")
                time.sleep(30)
    
    def _monitoring_loop(self):
        """Цикл мониторинга"""
        while self.is_running:
            try:
                self.risk_manager.monitor_risk()
                self.trading_engine.monitor_positions()
                time.sleep(60)  # Мониторинг каждую минуту
            except Exception as e:
                self.logger.error(f"{self.lang_manager.get_text('monitoring_error')}: {e}")
                time.sleep(60)
    
    def shutdown(self):
        """Корректное завершение работы"""
        self.logger.info(self.lang_manager.get_text('shutting_down'))
        self.is_running = False
        
        # Ожидание завершения потоков
        for thread in self.threads:
            thread.join(timeout=5)
        
        self.trading_engine.shutdown()
        self.data_handler.shutdown()
        
        self.logger.info(self.lang_manager.get_text('shutdown_complete'))

# ===== ГЛАВНАЯ ФУНКЦИЯ =====

def main():
    """Основная функция запуска"""
    # Загрузка переменных окружения
    load_dotenv()
    
    # Парсинг аргументов командной строки
    parser = argparse.ArgumentParser(description='🤖 AI Crypto Trading Agent')
    parser.add_argument('--language', '-l', default='en', choices=['en', 'ru'], 
                       help='Language: en, ru')
    parser.add_argument('--mode', '-m', default='paper', choices=['paper', 'live'], 
                       help='Trading mode: paper, live')
    parser.add_argument('--exchange', '-e', default='binance', 
                       help='Exchange: binance, bybit, kucoin, etc.')
    
    args = parser.parse_args()
    
    # Инициализация менеджера языков
    lang_manager = LanguageManager(args.language)
    
    # Настройка логгера
    logger = setup_logger('main', args.language)
    
    logger.info(lang_manager.get_text('welcome'))
    logger.info(lang_manager.get_text('initializing_agent'))
    
    try:
        # Создание и запуск агента
        agent = CryptoTradingAgent(
            language=args.language,
            mode=args.mode,
            exchange=args.exchange
        )
        
        agent.initialize()
        agent.run()
        
    except KeyboardInterrupt:
        logger.info(lang_manager.get_text('shutdown'))
        if 'agent' in locals():
            agent.shutdown()
    except Exception as e:
        logger.error(f"{lang_manager.get_text('error')}: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()