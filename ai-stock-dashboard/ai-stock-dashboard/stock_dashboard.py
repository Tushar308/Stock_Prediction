import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import streamlit as st
from datetime import datetime, timedelta
import joblib
import os
from analyzer import StockAnalyzer
import warnings
warnings.filterwarnings('ignore')

# Configure Streamlit page
st.set_page_config(
    page_title="AI Stock Market Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

def create_advanced_chart(data, symbol):
    """Create advanced candlestick chart with technical indicators"""
    fig = make_subplots(
        rows=4, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        subplot_titles=(f'{symbol} Price Action & Moving Averages', 'Volume', 'MACD', 'RSI & Stochastic'),
        row_heights=[0.5, 0.15, 0.2, 0.15]
    )
    
    # Candlestick chart
    fig.add_trace(
        go.Candlestick(
            x=data.index,
            open=data['Open'],
            high=data['High'],
            low=data['Low'],
            close=data['Close'],
            name='Price',
            increasing_line_color='#00ff88',
            decreasing_line_color='#ff4444'
        ),
        row=1, col=1
    )
    
    # Moving averages
    colors = ['#ff9500', '#007aff', '#5856d6']
    mas = [('SMA_20', 'SMA 20'), ('SMA_50', 'SMA 50'), ('SMA_200', 'SMA 200')]
    
    for i, (ma_col, ma_name) in enumerate(mas):
        if ma_col in data.columns and not data[ma_col].isna().all():
            fig.add_trace(
                go.Scatter(x=data.index, y=data[ma_col], 
                          line=dict(color=colors[i], width=1.5), name=ma_name),
                row=1, col=1
            )
    
    # Bollinger Bands
    if all(col in data.columns for col in ['BB_upper', 'BB_lower']):
        fig.add_trace(
            go.Scatter(x=data.index, y=data['BB_upper'], 
                      line=dict(color='rgba(128,128,128,0.5)', width=1), name='BB Upper',
                      showlegend=False),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=data.index, y=data['BB_lower'], 
                      line=dict(color='rgba(128,128,128,0.5)', width=1), name='BB Lower',
                      fill='tonexty', fillcolor='rgba(128,128,128,0.1)',
                      showlegend=False),
            row=1, col=1
        )
    
    # Volume
    volume_colors = ['#00ff88' if data['Close'].iloc[i] >= data['Open'].iloc[i] else '#ff4444' 
                    for i in range(len(data))]
    fig.add_trace(
        go.Bar(x=data.index, y=data['Volume'], 
              marker_color=volume_colors, name='Volume', opacity=0.7),
        row=2, col=1
    )
    
    if 'Volume_SMA' in data.columns:
        fig.add_trace(
            go.Scatter(x=data.index, y=data['Volume_SMA'], 
                      line=dict(color='white', width=1), name='Vol SMA'),
            row=2, col=1
        )
    
    # MACD
    if all(col in data.columns for col in ['MACD', 'MACD_signal', 'MACD_histogram']):
        fig.add_trace(
            go.Scatter(x=data.index, y=data['MACD'], 
                      line=dict(color='#007aff', width=2), name='MACD'),
            row=3, col=1
        )
        fig.add_trace(
            go.Scatter(x=data.index, y=data['MACD_signal'], 
                      line=dict(color='#ff9500', width=2), name='Signal'),
            row=3, col=1
        )
        
        histogram_colors = ['#00ff88' if val >= 0 else '#ff4444' for val in data['MACD_histogram']]
        fig.add_trace(
            go.Bar(x=data.index, y=data['MACD_histogram'], 
                  marker_color=histogram_colors, name='Histogram', opacity=0.6),
            row=3, col=1
        )
    
    # RSI and Stochastic
    if 'RSI' in data.columns:
        fig.add_trace(
            go.Scatter(x=data.index, y=data['RSI'], 
                      line=dict(color='#af52de', width=2), name='RSI'),
            row=4, col=1
        )
        # RSI levels
        fig.add_hline(y=70, line_dash="dash", line_color="red", opacity=0.7, row=4, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", opacity=0.7, row=4, col=1)
        fig.add_hline(y=50, line_dash="dot", line_color="gray", opacity=0.5, row=4, col=1)
    
    if 'Stoch_K' in data.columns:
        fig.add_trace(
            go.Scatter(x=data.index, y=data['Stoch_K'], 
                      line=dict(color='#ffcc00', width=1.5), name='Stoch %K'),
            row=4, col=1
        )
        fig.add_trace(
            go.Scatter(x=data.index, y=data['Stoch_D'], 
                      line=dict(color='#ff6600', width=1.5), name='Stoch %D'),
            row=4, col=1
        )
    
    fig.update_layout(
        title=f'{symbol} - Complete Technical Analysis Dashboard',
        xaxis_rangeslider_visible=False,
        height=900,
        showlegend=True,
        template='plotly_dark',
        font=dict(size=10)
    )
    
    # Remove x-axis labels from all but bottom subplot
    for i in range(1, 4):
        fig.update_xaxes(showticklabels=False, row=i, col=1)
    
    return fig

def create_performance_metrics(data, symbol):
    """Create performance metrics visualization"""
    # Calculate returns
    data['Daily_Returns'] = data['Close'].pct_change()
    data['Cumulative_Returns'] = (1 + data['Daily_Returns']).cumprod() - 1
    
    # Performance metrics
    total_return = data['Cumulative_Returns'].iloc[-1] * 100
    volatility = data['Daily_Returns'].std() * np.sqrt(252) * 100  # Annualized
    sharpe_ratio = (data['Daily_Returns'].mean() * 252) / (data['Daily_Returns'].std() * np.sqrt(252))
    
    max_drawdown = ((data['Close'] / data['Close'].expanding().max()) - 1).min() * 100
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Return", f"{total_return:.1f}%")
    with col2:
        st.metric("Volatility (Ann.)", f"{volatility:.1f}%")
    with col3:
        st.metric("Sharpe Ratio", f"{sharpe_ratio:.2f}")
    with col4:
        st.metric("Max Drawdown", f"{max_drawdown:.1f}%")
    
    # Cumulative returns chart
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=data.index,
            y=data['Cumulative_Returns'] * 100,
            mode='lines',
            name='Cumulative Returns',
            line=dict(color='#00ff88', width=2)
        )
    )
    
    fig.update_layout(
        title=f'{symbol} Cumulative Returns (%)',
        xaxis_title='Date',
        yaxis_title='Cumulative Return (%)',
        template='plotly_dark',
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)

# Streamlit App
def main():
    st.title("🚀 Professional AI Stock Market Dashboard")
    st.markdown("*Advanced technical analysis with machine learning predictions*")
    
    # Sidebar
    st.sidebar.header("📊 Dashboard Controls")
    st.sidebar.markdown("---")
    
    # Stock selection with popular choices
    popular_stocks = {
        'Apple': 'AAPL', 'Microsoft': 'MSFT', 'Google': 'GOOGL', 
        'Amazon': 'AMZN', 'Tesla': 'TSLA', 'NVIDIA': 'NVDA',
        'Meta': 'META', 'Netflix': 'NFLX', 'AMD': 'AMD', 'Intel': 'INTC'
    }
    
    stock_choice = st.sidebar.selectbox(
        "🏢 Select Stock:",
        options=list(popular_stocks.keys()) + ['Custom'],
        index=0
    )
    
    if stock_choice == 'Custom':
        symbol = st.sidebar.text_input("Enter Stock Symbol:", value="AAPL", max_chars=10).upper()
    else:
        symbol = popular_stocks[stock_choice]
    
    # Time period
    period = st.sidebar.selectbox(
        "📅 Analysis Period:",
        options=['1mo', '3mo', '6mo', '1y', '2y', '5y'],
        index=3
    )
    
    st.sidebar.markdown("---")
    
    # Analysis options
    st.sidebar.subheader("🔧 Analysis Options")
    show_prediction = st.sidebar.checkbox("🔮 ML Price Prediction", value=True)
    show_technical = st.sidebar.checkbox("📈 Technical Charts", value=True)
    show_performance = st.sidebar.checkbox("📊 Performance Metrics", value=True)
    show_analysis = st.sidebar.checkbox("🧠 AI Market Analysis", value=True)
    
    st.sidebar.markdown("---")
    
    if st.sidebar.button("🔄 Refresh Data", type="primary"):
        st.cache_data.clear()
        st.rerun()
    
    # Initialize analyzer
    analyzer = StockAnalyzer()
    model_dir = 'models'
    os.makedirs(model_dir, exist_ok=True)
    
    # Fetch and display data
    with st.spinner(f"📡 Fetching live data for {symbol}..."):
        data, info = analyzer.fetch_stock_data(symbol, period)
    
    if data is None or data.empty:
        st.error(f"❌ Could not fetch data for {symbol}. Please verify the symbol and try again.")
        st.info("💡 Try popular symbols like AAPL, MSFT, GOOGL, TSLA, etc.")
        return
    
    # Calculate technical indicators
    with st.spinner("⚙️ Calculating technical indicators..."):
        data = analyzer.calculate_technical_indicators(data)
    
    # Main dashboard header
    st.markdown("---")
    
    # Key metrics row
    col1, col2, col3, col4, col5 = st.columns(5)
    
    latest_price = data['Close'].iloc[-1]
    prev_price = data['Close'].iloc[-2]
    price_change = latest_price - prev_price
    price_change_pct = (price_change / prev_price) * 100
    
    with col1:
        st.metric(
            label="💰 Current Price",
            value=f"${latest_price:.2f}",
            delta=f"{price_change:.2f} ({price_change_pct:+.2f}%)"
        )
    
    with col2:
        volume = data['Volume'].iloc[-1]
        avg_volume = data['Volume'].rolling(20).mean().iloc[-1]
        volume_change = ((volume - avg_volume) / avg_volume) * 100 if avg_volume > 0 else 0
        st.metric(
            label="📊 Volume",
            value=f"{volume:,.0f}",
            delta=f"{volume_change:+.1f}% vs 20d avg"
        )
    
    with col3:
        if 'RSI' in data.columns and not pd.isna(data['RSI'].iloc[-1]):
            rsi = data['RSI'].iloc[-1]
            rsi_status = "Overbought" if rsi > 70 else "Oversold" if rsi < 30 else "Neutral"
            st.metric(
                label="⚡ RSI (14)",
                value=f"{rsi:.1f}",
                delta=rsi_status
            )
        else:
            st.metric(label="⚡ RSI (14)", value="N/A")
    
    with col4:
        if 'SMA_20' in data.columns and not pd.isna(data['SMA_20'].iloc[-1]):
            sma_20 = data['SMA_20'].iloc[-1]
            sma_distance = ((latest_price - sma_20) / sma_20) * 100
            st.metric(
                label="📈 vs SMA 20",
                value=f"{sma_distance:+.1f}%",
                delta="Above" if sma_distance > 0 else "Below"
            )
        else:
            st.metric(label="📈 vs SMA 20", value="N/A")
    
    with col5:
        market_cap = info.get('marketCap', 0)
        if market_cap:
            if market_cap > 1e12:
                cap_display = f"${market_cap/1e12:.2f}T"
            elif market_cap > 1e9:
                cap_display = f"${market_cap/1e9:.1f}B"
            else:
                cap_display = f"${market_cap/1e6:.0f}M"
            st.metric(label="🏢 Market Cap", value=cap_display)
        else:
            st.metric(label="🏢 Market Cap", value="N/A")
    
    st.markdown("---")
    
    # Advanced Chart
    if show_technical:
        st.subheader("📈 Advanced Technical Analysis")
        with st.spinner("Creating advanced charts..."):
            chart = create_advanced_chart(data, symbol)
            st.plotly_chart(chart, use_container_width=True)
    
    # Performance Metrics
    if show_performance:
        st.subheader("📊 Performance Analysis")
        create_performance_metrics(data, symbol)
    
    # ML Prediction
    if show_prediction:
        st.subheader("🔮 Machine Learning Price Prediction")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
                # Try to load a saved model
                model_info = None
                model_path = os.path.join(model_dir, f"{symbol}_model.joblib")
                if os.path.exists(model_path):
                    try:
                        saved = joblib.load(model_path)
                        analyzer.model = saved.get('model', analyzer.model)
                        analyzer.scaler = saved.get('scaler', analyzer.scaler)
                        feature_cols = saved.get('feature_cols')
                        # Recreate last_features from data
                        with st.spinner('Loading saved model...'):
                            data_ml = analyzer.prepare_ml_features(data)
                            data_ml = data_ml.dropna()
                            if feature_cols and all(col in data_ml.columns for col in feature_cols):
                                last_feats = data_ml[feature_cols].fillna(method='ffill').fillna(method='bfill').iloc[-1:]
                                model_info = {'last_features': last_feats, 'feature_cols': feature_cols}
                    except Exception as e:
                        st.warning(f"Could not load saved model: {e}")

                if model_info is None:
                    with st.spinner("🤖 Training AI prediction model..."):
                        model_info = analyzer.train_prediction_model(data)

                    # Save model for reuse
                    if model_info:
                        try:
                            joblib.dump({'model': analyzer.model, 'scaler': analyzer.scaler, 'feature_cols': model_info['feature_cols']}, model_path)
                        except Exception:
                            pass

                if model_info:
                    prediction = analyzer.predict_next_price(model_info)
                current_price = data['Close'].iloc[-1]
                predicted_change = ((prediction - current_price) / current_price) * 100
                
                st.success("✅ Model trained successfully!")
                
                pred_col1, pred_col2 = st.columns(2)
                with pred_col1:
                    st.metric(
                        label="🎯 Next Day Prediction",
                        value=f"${prediction:.2f}",
                        delta=f"{predicted_change:+.2f}%"
                    )
                
                with pred_col2:
                    confidence = model_info['test_score']
                    confidence_level = "High" if confidence > 0.8 else "Medium" if confidence > 0.6 else "Low"
                    st.metric(
                        label="🎲 Model Confidence",
                        value=f"{confidence:.1%}",
                        delta=confidence_level
                    )
                
                # Model performance
                st.info(f"📈 **Training Accuracy:** {model_info['train_score']:.1%} | **Test Accuracy:** {model_info['test_score']:.1%}")
            else:
                st.warning("⚠️ Insufficient data for reliable ML prediction. Need more historical data.")
        
        with col2:
            if model_info:
                # Feature importance
                importance_df = pd.DataFrame(
                    list(model_info['feature_importance'].items()),
                    columns=['Feature', 'Importance']
                ).sort_values('Importance', ascending=False).head(10)
                
                fig_importance = px.bar(
                    importance_df, 
                    x='Importance', 
                    y='Feature',
                    orientation='h',
                    title="🔍 Top 10 Most Important Features",
                    template='plotly_dark'
                )
                fig_importance.update_layout(height=400)
                st.plotly_chart(fig_importance, use_container_width=True)
    
    # AI Market Analysis
    if show_analysis:
        st.subheader("🧠 AI-Powered Market Analysis")
        
        with st.spinner("🤖 Generating intelligent market insights..."):
            analysis = analyzer.generate_market_analysis(data, info, symbol)
        
        # Display analysis in an attractive format
        for i, insight in enumerate(analysis):
            if i == 0:  # First insight (price movement) gets special treatment
                if "🚀" in insight or "🟢" in insight:
                    st.success(insight)
                elif "🔴" in insight or "🔻" in insight:
                    st.error(insight)
                else:
                    st.warning(insight)
            else:
                st.info(insight)
    
    # Additional Analysis Tabs
    st.markdown("---")
    
    tab1, tab2, tab3 = st.tabs(["📋 Company Info", "📊 Raw Data", "🔧 Technical Indicators"])
    
    with tab1:
        if info:
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("### 🏢 Company Details")
                company_info = {
                    "Company Name": info.get('longName', 'N/A'),
                    "Sector": info.get('sector', 'N/A'),
                    "Industry": info.get('industry', 'N/A'),
                    "Country": info.get('country', 'N/A'),
                    "Website": info.get('website', 'N/A'),
                    "Employees": f"{info.get('fullTimeEmployees', 'N/A'):,}" if info.get('fullTimeEmployees') else 'N/A'
                }
                
                for key, value in company_info.items():
                    st.write(f"**{key}:** {value}")
            
            with col2:
                st.write("### 📈 Financial Metrics")
                financial_info = {
                    "P/E Ratio": f"{info.get('trailingPE', 'N/A'):.2f}" if info.get('trailingPE') else 'N/A',
                    "Forward P/E": f"{info.get('forwardPE', 'N/A'):.2f}" if info.get('forwardPE') else 'N/A',
                    "PEG Ratio": f"{info.get('pegRatio', 'N/A'):.2f}" if info.get('pegRatio') else 'N/A',
                    "Price to Book": f"{info.get('priceToBook', 'N/A'):.2f}" if info.get('priceToBook') else 'N/A',
                    "Dividend Yield": f"{info.get('dividendYield', 0)*100:.2f}%" if info.get('dividendYield') else 'N/A',
                    "Beta": f"{info.get('beta', 'N/A'):.2f}" if info.get('beta') else 'N/A',
                    "52W High": f"${info.get('fiftyTwoWeekHigh', 'N/A'):.2f}" if info.get('fiftyTwoWeekHigh') else 'N/A',
                    "52W Low": f"${info.get('fiftyTwoWeekLow', 'N/A'):.2f}" if info.get('fiftyTwoWeekLow') else 'N/A'
                }
                
                for key, value in financial_info.items():
                    st.write(f"**{key}:** {value}")
        else:
            st.warning("Company information not available")
    
    with tab2:
        st.write("### 📊 Recent Price Data")
        display_data = data[['Open', 'High', 'Low', 'Close', 'Volume']].tail(20)
        display_data.index = display_data.index.strftime('%Y-%m-%d')
        st.dataframe(display_data, use_container_width=True)
        
        # Download option
        csv = display_data.to_csv()
        st.download_button(
            label="📥 Download Data as CSV",
            data=csv,
            file_name=f'{symbol}_stock_data.csv',
            mime='text/csv'
        )
    
    with tab3:
        st.write("### 🔧 Technical Indicators (Last 10 Days)")
        
        tech_columns = ['Close', 'SMA_20', 'SMA_50', 'RSI', 'MACD', 'MACD_signal', 'BB_upper', 'BB_lower', 'ATR']
        available_columns = [col for col in tech_columns if col in data.columns]
        
        if available_columns:
            tech_data = data[available_columns].tail(10)
            tech_data.index = tech_data.index.strftime('%Y-%m-%d')
            st.dataframe(tech_data.round(3), use_container_width=True)
        else:
            st.warning("Technical indicators not available")
    
    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: #666; padding: 20px;'>
            <p>🚀 <strong>AI Stock Dashboard</strong> - Professional technical analysis with machine learning</p>
            <p><em>⚠️ This is for educational purposes only. Not financial advice.</em></p>
            <p>Built with ❤️ by <a href='https://erikthiart.com' target='_blank'>Erik Thiart</a></p>
            <p>📊 Powered by <a href='https://plotly.com' target='_blank'>Plotly</a> and <a href='https://streamlit.io' target='_blank'>Streamlit</a></p>
        </div>
        """, 
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()