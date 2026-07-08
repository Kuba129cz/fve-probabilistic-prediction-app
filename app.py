from datetime import date, timedelta
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from src import data_processing
from src.predict import predict

st.set_page_config(page_title="PV Prediction - Data", layout="wide")
st.title("PV Probabilistic Prediction App")


def render_forecast_dashboard(predictions, time_axis, selected_date, actual_energy=None):
    """
    This function handles converting predictions into a DataFrame and rendering the interactive chart.
    If actual_energy list is provided, it adds it to the chart as a dashed line representing reality.
    """
    st.markdown("---")
    st.subheader("Interactive PV Production Prediction Chart")

    quantile_names = ['10% Quantile', '30% Quantile', 'Median (50%)', '70% Quantile', '90% Quantile']
    df_preds = pd.DataFrame(predictions, columns=quantile_names, index=time_axis)

    fig = go.Figure()

    # 10% - 90% Uncertainty band (outer lighter area)
    fig.add_trace(go.Scatter(
        x=df_preds.index, y=df_preds['90% Quantile'],
        mode='lines', line=dict(width=0), showlegend=False
    ))
    fig.add_trace(go.Scatter(
        x=df_preds.index, y=df_preds['10% Quantile'],
        mode='lines', line=dict(width=0),
        fill='tonexty', fillcolor='rgba(255, 165, 0, 0.15)',
        name='10% - 90% Confidence Interval'
    ))

    # 30% - 70% Uncertainty band (inner darker area)
    fig.add_trace(go.Scatter(
        x=df_preds.index, y=df_preds['70% Quantile'],
        mode='lines', line=dict(width=0), showlegend=False
    ))
    fig.add_trace(go.Scatter(
        x=df_preds.index, y=df_preds['30% Quantile'],
        mode='lines', line=dict(width=0),
        fill='tonexty', fillcolor='rgba(255, 165, 0, 0.3)',
        name='30% - 70% Confidence Interval'
    ))

    # Main prediction line - Median (50%)
    fig.add_trace(go.Scatter(
        x=df_preds.index, y=df_preds['Median (50%)'],
        mode='lines+markers',
        line=dict(color='rgb(255, 120, 0)', width=3),
        name='Expected Production (Median)'
    ))

    # If historical actual data is available, plot it as a blue dashed line
    if actual_energy is not None:
        fig.add_trace(go.Scatter(
            x=df_preds.index, y=actual_energy,
            mode='lines+markers',
            line=dict(color='rgb(31, 119, 180)', width=3, dash='dash'),
            name='Actual Production (Reality)'
        ))

    # Chart layout configuration
    fig.update_layout(
        title=f"Probabilistic Forecast vs Reality for {selected_date}",
        xaxis_title="Hour",
        yaxis_title="PV Production (kW / your units)",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        template="plotly_white"
    )

    st.plotly_chart(fig, use_container_width='stretch')

    if actual_energy is not None:
        df_preds.insert(0, 'Actual (Reality)', actual_energy)

    with st.expander("👁️ View predicted data in a table"):
        st.dataframe(df_preds, use_container_width='stretch')


def main():
    today = date.today()

    with st.form(key="prediction_form"):
        selected_date = st.date_input(label="Select a date", value=today, max_value=today)
        submit_button = st.form_submit_button(label="Predict")

    if submit_button:
        st.markdown(f"**Selected date:** `{selected_date}` (History is being fetched for `{selected_date - timedelta(days=1)}`)")
        st.markdown("---")

        with st.spinner("Loading and transforming data from API..."):
            result = data_processing.get_meteo_dataset(predicted_day=selected_date)
        
        if result is None:
            st.error("Error: Could not load data for this date.")
            return
            
        df_past, df_future, actual_energy = result

        st.subheader(f"Historical data (Weather + PV production for {selected_date - timedelta(days=1)})")
        if df_past is not None and not df_past.empty:
            st.dataframe(df_past, use_container_width='stretch')
        else:
            st.warning("Historical data is not available.")

        st.markdown("---")

        st.subheader(f"Future data (Weather forecast for {selected_date})")
        if df_future is not None and not df_future.empty:
            st.dataframe(df_future, use_container_width='stretch')
        else:
            st.warning("Weather forecast is not available.")
        
        if df_future is not None and 'time' in df_future.columns:
            time_axis = pd.to_datetime(df_future['time']).dt.strftime('%H:%M').tolist()
        else:
            time_axis = [f"{p:02d}:00" for p in range(24)]

        with st.spinner("Generating probabilistic forecast..."):
            predictions = predict(x_past=df_past, x_future=df_future)
        
        render_forecast_dashboard(
            predictions=predictions, 
            time_axis=time_axis, 
            selected_date=selected_date,
            actual_energy=actual_energy
        )


if __name__ == "__main__":
    main()