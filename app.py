import streamlit as st
import pandas as pd
import numpy as np

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    accuracy_score,
    classification_report,
)

from mlxtend.frequent_patterns import apriori, association_rules

# ---------------------------------------------------
# ⚙️ Streamlit basic config
# ---------------------------------------------------
st.set_page_config(
    page_title="SuperStore Analytics",
    layout="wide",
    initial_sidebar_state="expanded",
)

px.defaults.template = "plotly_dark"

# ---------------------------------------------------
# 📥 Data loading & preprocessing
# ---------------------------------------------------

@st.cache_data
def load_data(uploaded_file=None, path_fallback="SuperStoreOrdersOG.csv"):
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_csv(path_fallback)
    return df


@st.cache_data
def preprocess_data(data_raw: pd.DataFrame):
    data = data_raw.copy()

    # Convert dates
    for col in ["order_date", "ship_date"]:
        data[col] = pd.to_datetime(data[col], errors="coerce", dayfirst=True)

    # Fix numeric columns
    data["sales"] = pd.to_numeric(data["sales"], errors="coerce")

    # Drop rows where key targets are missing
    data = data.dropna(subset=["sales", "profit"])

    # Separate numeric and categorical
    num_cols = data.select_dtypes(include=["int64", "float64"]).columns.tolist()
    cat_cols = data.select_dtypes(include=["object"]).columns.tolist()

    # Fill numeric NaNs with median
    for col in num_cols:
        data[col] = data[col].fillna(data[col].median())

    # Fill categorical NaNs with mode
    for col in cat_cols:
        if data[col].isna().sum() > 0:
            data[col] = data[col].fillna(data[col].mode()[0])

    # Feature engineering
    data["shipping_delay_days"] = (
        data["ship_date"] - data["order_date"]
    ).dt.days

    data["order_year"] = data["order_date"].dt.year
    data["order_month"] = data["order_date"].dt.month
    data["order_quarter"] = data["order_date"].dt.quarter

    data["profit_margin"] = np.where(
        data["sales"] != 0, data["profit"] / data["sales"], 0
    )

    data["is_discounted"] = (data["discount"] > 0).astype(int)
    data["is_profitable"] = (data["profit"] > 0).astype(int)
    data["high_sales_flag"] = (
        data["sales"] > data["sales"].median()
    ).astype(int)

    return data


@st.cache_data
def train_regression_model(data: pd.DataFrame):
    features = [
        "sales",
        "quantity",
        "discount",
        "shipping_cost",
        "shipping_delay_days",
        "order_year",
        "order_month",
        "profit_margin",
        "segment",
        "category",
        "sub_category",
        "region",
        "market",
        "order_priority",
    ]

    model_data = data[features + ["profit"]].dropna()
    model_data = pd.get_dummies(
        model_data,
        columns=[
            "segment",
            "category",
            "sub_category",
            "region",
            "market",
            "order_priority",
        ],
        drop_first=True,
    )

    X = model_data.drop("profit", axis=1)
    y = model_data["profit"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestRegressor(
        n_estimators=150, random_state=42, n_jobs=-1
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = mse ** 0.5
    r2 = r2_score(y_test, y_pred)

    feature_importances = pd.DataFrame(
        {"feature": X.columns, "importance": model.feature_importances_}
    ).sort_values("importance", ascending=False)

    return model, (mae, rmse, r2), feature_importances


@st.cache_data
def train_classification_model(data: pd.DataFrame):
    features = [
        "sales",
        "quantity",
        "discount",
        "shipping_cost",
        "shipping_delay_days",
        "order_year",
        "order_month",
        "segment",
        "category",
        "sub_category",
        "region",
        "market",
        "order_priority",
    ]

    clf_data = data[features + ["is_profitable"]].dropna()
    clf_data = pd.get_dummies(
        clf_data,
        columns=[
            "segment",
            "category",
            "sub_category",
            "region",
            "market",
            "order_priority",
        ],
        drop_first=True,
    )

    X = clf_data.drop("is_profitable", axis=1)
    y = clf_data["is_profitable"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    model = RandomForestClassifier(
        n_estimators=200, random_state=42, n_jobs=-1
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    report_df = pd.DataFrame(report).transpose()

    feature_importances = pd.DataFrame(
        {"feature": X.columns, "importance": model.feature_importances_}
    ).sort_values("importance", ascending=False)

    return model, acc, report_df, feature_importances


@st.cache_data
def compute_association_rules(
    data: pd.DataFrame, min_support: float = 0.003
):
    # use sub_category-level baskets
    basket = (
        data.groupby(["order_id", "sub_category"])["quantity"]
        .sum()
        .unstack()
        .fillna(0)
    )

    basket_binary = basket.applymap(lambda x: 1 if x > 0 else 0)
    # keep only orders with >1 item
    basket_binary = basket_binary[basket_binary.sum(axis=1) > 1]

    frequent_itemsets = apriori(
        basket_binary,
        min_support=min_support,
        use_colnames=True,
    )

    if frequent_itemsets.empty:
        return frequent_itemsets, pd.DataFrame()

    rules = association_rules(
        frequent_itemsets, metric="lift", min_threshold=1.0
    )

    rules = rules.sort_values(
        ["confidence", "lift"], ascending=False
    ).reset_index(drop=True)

    return frequent_itemsets, rules


# ---------------------------------------------------
# 🎛️ Sidebar – controls
# ---------------------------------------------------
st.sidebar.title("⚙️ Settings")

uploaded_file = st.sidebar.file_uploader(
    "Upload SuperStoreOrdersOG.csv",
    type=["csv"],
    help="If you don't upload, the app will try to load SuperStoreOrdersOG.csv from the repo.",
)

min_support = st.sidebar.slider(
    "Min support for association rules (sub-category level)",
    min_value=0.0005,
    max_value=0.02,
    value=0.003,
    step=0.0005,
)

st.sidebar.markdown("---")
st.sidebar.markdown("Made for your **Data Analysis Project** 💚")

# ---------------------------------------------------
# 📥 Load + preprocess
# ---------------------------------------------------
try:
    raw_data = load_data(uploaded_file)
except Exception as e:
    st.error(
        "Could not load data. Upload a CSV or ensure SuperStoreOrdersOG.csv exists in the repo."
    )
    st.stop()

data = preprocess_data(raw_data)

# ---------------------------------------------------
# 🧷 Main layout
# ---------------------------------------------------
st.title("📊 SuperStore Analytics – Streamlit App")

st.caption(
    "End-to-end project: preprocessing, regression, classification, association rules & interactive dashboard."
)

tab_overview, tab_dash, tab_reg, tab_clf, tab_assoc = st.tabs(
    ["🔍 Overview", "📈 Dashboard", "📉 Regression", "🎯 Classification", "🛒 Association Rules"]
)

# ---------------------------------------------------
# 🔍 OVERVIEW TAB
# ---------------------------------------------------
with tab_overview:
    st.subheader("Dataset Snapshot")
    st.write(f"Rows: `{data.shape[0]}`, Columns: `{data.shape[1]}`")

    st.dataframe(data.head())

    st.markdown("### Column Types")
    col_types = pd.DataFrame(
        {
            "column": data.columns,
            "dtype": data.dtypes.astype(str),
        }
    )
    st.dataframe(col_types, use_container_width=True)

    st.markdown("### Basic Summary")
    st.dataframe(data.describe(include="all").transpose())

# ---------------------------------------------------
# 📈 DASHBOARD TAB
# ---------------------------------------------------
with tab_dash:
    st.subheader("Executive Dashboard")

    # Filters
    col1, col2, col3 = st.columns(3)
    with col1:
        year_filter = st.multiselect(
            "Year", sorted(data["order_year"].dropna().unique()),
            default=sorted(data["order_year"].dropna().unique())
        )
    with col2:
        segment_filter = st.multiselect(
            "Segment", sorted(data["segment"].unique()),
            default=sorted(data["segment"].unique())
        )
    with col3:
        region_filter = st.multiselect(
            "Region", sorted(data["region"].unique()),
            default=sorted(data["region"].unique())
        )

    df_f = data[
        data["order_year"].isin(year_filter)
        & data["segment"].isin(segment_filter)
        & data["region"].isin(region_filter)
    ]

    # KPIs
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Total Sales", f"{df_f['sales'].sum():,.0f}")
    with c2:
        st.metric("Total Profit", f"{df_f['profit'].sum():,.0f}")
    with c3:
        st.metric("Avg Profit Margin", f"{df_f['profit_margin'].mean():.2%}")
    with c4:
        st.metric("Orders", f"{df_f['order_id'].nunique():,}")

    # 2x2 grid of plots
    fig_dash = make_subplots(
        rows=2,
        cols=2,
        specs=[
            [{"type": "scatter"}, {"type": "choropleth"}],
            [{"type": "scatter"}, {"type": "domain"}],
        ],
        subplot_titles=[
            "Monthly Sales Trend",
            "Profit by Country",
            "Sales vs Profit (Bubble)",
            "Category → Sub-category Sales (Sunburst)",
        ],
    )

    # 1) Monthly sales trend
    monthly_sales = (
        df_f.groupby(pd.Grouper(key="order_date", freq="M"))["sales"]
        .sum()
        .reset_index()
    )
    fig_dash.add_trace(
        go.Scatter(
            x=monthly_sales["order_date"],
            y=monthly_sales["sales"],
            mode="lines+markers",
            name="Monthly Sales",
        ),
        row=1,
        col=1,
    )

    # 2) Map – profit by country
    profit_country = (
        df_f.groupby("country")["profit"].sum().reset_index()
    )
    if not profit_country.empty:
        fig_dash.add_trace(
            go.Choropleth(
                locations=profit_country["country"],
                locationmode="country names",
                z=profit_country["profit"],
                colorscale="Viridis",
                colorbar_title="Profit",
                name="Profit by Country",
            ),
            row=1,
            col=2,
        )

    # 3) Sales vs profit bubble
    fig_dash.add_trace(
        go.Scatter(
            x=df_f["sales"],
            y=df_f["profit"],
            mode="markers",
            marker=dict(
                size=np.clip(df_f["quantity"] * 2, 4, 30),
                sizemode="area",
                opacity=0.6,
            ),
            text=df_f["segment"],
            name="Sales vs Profit",
        ),
        row=2,
        col=1,
    )

    # 4) Sunburst – category → sub_category
    sunburst_values = (
        df_f.groupby(["category", "sub_category"])["sales"]
        .sum()
        .reset_index()
    )
    if not sunburst_values.empty:
        fig_dash.add_trace(
            go.Sunburst(
                labels=sunburst_values["sub_category"],
                parents=sunburst_values["category"],
                values=sunburst_values["sales"],
                branchvalues="total",
                name="Sunburst",
            ),
            row=2,
            col=2,
        )

    fig_dash.update_layout(
        height=800,
        template="plotly_dark",
        showlegend=False,
        margin=dict(t=60, l=20, r=20, b=20),
    )

    st.plotly_chart(fig_dash, use_container_width=True)

# ---------------------------------------------------
# 📉 REGRESSION TAB
# ---------------------------------------------------
with tab_reg:
    st.subheader("Regression – Predict Profit")

    with st.spinner("Training regression model..."):
        reg_model, (mae, rmse, r2), fi_reg = train_regression_model(
            data
        )

    c1, c2, c3 = st.columns(3)
    c1.metric("MAE", f"{mae:,.2f}")
    c2.metric("RMSE", f"{rmse:,.2f}")
    c3.metric("R²", f"{r2:.3f}")

    st.markdown("#### Top Features Influencing Profit")
    fi_top = fi_reg.head(20)
    fig_fi_reg = px.bar(
        fi_top,
        x="importance",
        y="feature",
        orientation="h",
        title="Feature Importance (Regression)",
    )
    fig_fi_reg.update_layout(yaxis={"categoryorder": "total ascending"})
    st.plotly_chart(fig_fi_reg, use_container_width=True)

# ---------------------------------------------------
# 🎯 CLASSIFICATION TAB
# ---------------------------------------------------
with tab_clf:
    st.subheader("Classification – Is Order Profitable? (0/1)")

    with st.spinner("Training classification model..."):
        (
            clf_model,
            acc,
            report_df,
            fi_clf,
        ) = train_classification_model(data)

    st.metric("Accuracy", f"{acc:.3f}")

    st.markdown("#### Classification Report")
    st.dataframe(report_df, use_container_width=True)

    st.markdown("#### Top Features Influencing Profitability")
    fi_top_clf = fi_clf.head(20)
    fig_fi_clf = px.bar(
        fi_top_clf,
        x="importance",
        y="feature",
        orientation="h",
        title="Feature Importance (Classification)",
    )
    fig_fi_clf.update_layout(yaxis={"categoryorder": "total ascending"})
    st.plotly_chart(fig_fi_clf, use_container_width=True)

# ---------------------------------------------------
# 🛒 ASSOCIATION RULES TAB
# ---------------------------------------------------
with tab_assoc:
    st.subheader("Association Rules – Sub-category Level")

    with st.spinner("Mining frequent itemsets & rules..."):
        frequent_itemsets, rules = compute_association_rules(
            data, min_support=min_support
        )

    st.write(f"Number of frequent itemsets: {frequent_itemsets.shape[0]}")
    st.write(f"Number of rules: {rules.shape[0]}")

    if rules.empty:
        st.warning(
            "No association rules found with current settings. "
            "Try lowering min support in the sidebar."
        )
    else:
        # Show top rules
        rules_show = rules.copy()
        rules_show["antecedents"] = rules_show["antecedents"].astype(str)
        rules_show["consequents"] = rules_show["consequents"].astype(str)

        st.markdown("#### Top 20 Rules (sorted by confidence & lift)")
        st.dataframe(
            rules_show[
                [
                    "antecedents",
                    "consequents",
                    "support",
                    "confidence",
                    "lift",
                ]
            ].head(20),
            use_container_width=True,
        )

        # Scatter plot: support vs confidence (size = lift)
        top_rules = rules_show.head(50)
        fig_rules = px.scatter(
            top_rules,
            x="support",
            y="confidence",
            size="lift",
            color="lift",
            hover_data=["antecedents", "consequents"],
            title="Top Association Rules (Sub-category Level)",
        )
        st.plotly_chart(fig_rules, use_container_width=True)
