import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Telecom Churn Prediction",
    page_icon="📱",
    layout="wide"
)


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(
    BASE_DIR,
    "models"
)


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_churn_model():

    model = joblib.load(
        os.path.join(
            MODEL_DIR,
            "churn_model_1month.pkl"
        )
    )

    imputer = joblib.load(
        os.path.join(
            MODEL_DIR,
            "imputer_1month.pkl"
        )
    )

    features = joblib.load(
        os.path.join(
            MODEL_DIR,
            "features_1month.pkl"
        )
    )

    thresholds = joblib.load(
        os.path.join(
            MODEL_DIR,
            "high_value_thresholds.pkl"
        )
    )

    return model, imputer, features, thresholds


# ============================================================
# LOAD
# ============================================================

try:

    model, imputer, model_features, thresholds = (
        load_churn_model()
    )

except Exception as e:

    st.error(
        "Unable to load the churn model files."
    )

    st.code(str(e))

    st.stop()


# ============================================================
# HEADER
# ============================================================

st.title("📱 Telecom Customer Churn Prediction")

st.write(
    "Predict next-month customer churn using "
    "current-month customer behaviour."
)

st.divider()


# ============================================================
# MONTH SELECTION
# ============================================================

st.sidebar.header("Prediction Settings")

months = [
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December"
]

month = st.sidebar.selectbox(
    "Current Customer Data Month",
    months
)

month_index = months.index(month)

next_month = months[
    (month_index + 1) % 12
]

st.sidebar.info(
    f"{month} behaviour → {next_month} churn"
)


# ============================================================
# FILE UPLOAD
# ============================================================

st.header(
    f"Upload {month} Customer Data"
)

uploaded_file = st.file_uploader(
    "Upload CSV",
    type=["csv"]
)


if uploaded_file is None:

    st.info(
        f"Upload the customer data for {month}. "
        f"The model will predict {next_month} churn."
    )

    st.markdown(
        """
        ### How the model works

        **Input**

        One month's customer behaviour.

        **Model**

        Saved Random Forest churn classifier.

        **Output**

        - Churn probability
        - Predicted churn
        - Risk segment
        - High-risk customers
        - ARPU / revenue exposure

        **Important**

        The selected month is used only for display.
        It is not passed to the machine-learning model.
        """
    )

    st.stop()


# ============================================================
# LOAD DATA
# ============================================================

try:

    df = pd.read_csv(
        uploaded_file
    )

except Exception as e:

    st.error(
        "Unable to read the uploaded CSV."
    )

    st.code(str(e))

    st.stop()


st.success(
    f"Successfully loaded {len(df):,} customers."
)


# ============================================================
# VALIDATE UPLOADED FILE MONTH
# ============================================================

filename = uploaded_file.name.lower()

month_keywords = {
    "January": ["jan", "january"],
    "February": ["feb", "february"],
    "March": ["mar", "march"],
    "April": ["apr", "april"],
    "May": ["may"],
    "June": ["jun", "june"],
    "July": ["jul", "july"],
    "August": ["aug", "august"],
    "September": ["sep", "sept", "september"],
    "October": ["oct", "october"],
    "November": ["nov", "november"],
    "December": ["dec", "december"]
}

expected_keywords = month_keywords[month]

if not any(
    keyword in filename
    for keyword in expected_keywords
):

    st.error(
        f"❌ Month mismatch: You selected {month}, "
        f"but uploaded '{uploaded_file.name}'."
    )

    st.warning(
        f"Please upload the {month} customer dataset."
    )

    st.stop()

# ============================================================
# PREVIEW
# ============================================================

with st.expander(
    "Preview uploaded data"
):

    st.dataframe(
        df.head(10),
        use_container_width=True
    )


# ============================================================
# DETERMINE CURRENT MONTH COLUMN
# ============================================================

month_numbers = {
    "January": "1",
    "February": "2",
    "March": "3",
    "April": "4",
    "May": "5",
    "June": "6",
    "July": "7",
    "August": "8",
    "September": "9",
    "October": "10",
    "November": "11",
    "December": "12"
}

month_number = month_numbers[month]

arpu_column = (
    f"arpu_{month_number}"
)


# ============================================================
# HIGH-VALUE THRESHOLD
# ============================================================

threshold = None

if isinstance(thresholds, dict):

    if month in thresholds:

        threshold = thresholds[month]

    elif month_number in thresholds:

        threshold = thresholds[month_number]


# ============================================================
# PREPARE DATA
# ============================================================

X_new = df.copy()


# ============================================================
# IDENTIFY MONTH-SPECIFIC COLUMNS
# ============================================================

current_suffix = f"_{month_number}"

current_month_columns = []

for col in X_new.columns:

    if col.endswith(current_suffix):

        current_month_columns.append(
            col
        )

    elif col in [
        "mobile_number",
        "circle_id",
        "aon"
    ]:

        current_month_columns.append(
            col
        )


# ============================================================
# HANDLE VBC 3G COLUMN
# ============================================================

vbc_columns = {
    "January": "jan_vbc_3g",
    "February": "feb_vbc_3g",
    "March": "mar_vbc_3g",
    "April": "apr_vbc_3g",
    "May": "may_vbc_3g",
    "June": "jun_vbc_3g",
    "July": "jul_vbc_3g",
    "August": "aug_vbc_3g",
    "September": "sep_vbc_3g",
    "October": "oct_vbc_3g",
    "November": "nov_vbc_3g",
    "December": "dec_vbc_3g"
}

vbc_column = vbc_columns.get(
    month
)

if (
    vbc_column is not None
    and vbc_column in X_new.columns
):

    current_month_columns.append(
        vbc_column
    )


# ============================================================
# KEEP ONLY CURRENT-MONTH DATA
# ============================================================

if current_month_columns:

    X_new = X_new[
        list(
            dict.fromkeys(
                current_month_columns
            )
        )
    ].copy()


# ============================================================
# RENAME MONTH-SPECIFIC FEATURES
# ============================================================

rename_dict = {}

for col in X_new.columns:

    if col.endswith(
        current_suffix
    ):

        new_name = col[
            :-len(current_suffix)
        ]

        rename_dict[col] = new_name


if (
    vbc_column is not None
    and vbc_column in X_new.columns
):

    rename_dict[
        vbc_column
    ] = "vbc_3g"


X_new = X_new.rename(
    columns=rename_dict
)


# ============================================================
# REMOVE IDENTIFIERS
# ============================================================

X_new = X_new.drop(
    columns=[
        "mobile_number"
    ],
    errors="ignore"
)


# ============================================================
# REMOVE DATE COLUMNS
# ============================================================

date_columns = [
    col
    for col in X_new.columns
    if "date" in col.lower()
]

X_new = X_new.drop(
    columns=date_columns,
    errors="ignore"
)


# ============================================================
# ONE-HOT ENCODING
# ============================================================

categorical_columns = (
    X_new
    .select_dtypes(
        include=[
            "object",
            "category"
        ]
    )
    .columns
    .tolist()
)

if categorical_columns:

    X_new = pd.get_dummies(
        X_new,
        columns=categorical_columns,
        drop_first=True
    )


# ============================================================
# ALIGN WITH TRAINING FEATURES
# ============================================================

for col in model_features:

    if col not in X_new.columns:

        X_new[col] = 0


X_new = X_new[
    model_features
]


# ============================================================
# NUMERIC CONVERSION
# ============================================================

X_new = X_new.apply(
    pd.to_numeric,
    errors="coerce"
)


# ============================================================
# IMPUTATION
# ============================================================

try:

    X_new = pd.DataFrame(
        imputer.transform(
            X_new
        ),
        columns=model_features,
        index=X_new.index
    )

except Exception as e:

    st.error(
        "Preprocessing failed."
    )

    st.code(str(e))

    st.stop()


# ============================================================
# PREDICTION
# ============================================================

try:

    probabilities = (
        model.predict_proba(
            X_new
        )[:, 1]
    )

except Exception as e:

    st.error(
        "Churn prediction failed."
    )

    st.code(str(e))

    st.stop()


# ============================================================
# ADD RESULTS
# ============================================================

result_df = df.copy()

result_df[
    "Churn_Probability"
] = probabilities

result_df[
    "Predicted_Churn"
] = (
    probabilities >= 0.5
).astype(int)


# ============================================================
# RISK SEGMENT
# ============================================================

result_df[
    "Risk_Segment"
] = pd.cut(
    result_df[
        "Churn_Probability"
    ],
    bins=[
        0,
        0.30,
        0.60,
        1.00
    ],
    labels=[
        "Low Risk",
        "Medium Risk",
        "High Risk"
    ],
    include_lowest=True
)


# ============================================================
# CURRENT ARPU
# ============================================================

if arpu_column in result_df.columns:

    result_df[
        "current_month_arpu"
    ] = pd.to_numeric(
        result_df[
            arpu_column
        ],
        errors="coerce"
    )


# ============================================================
# HEADER
# ============================================================

st.divider()

st.header(
    f"📊 {next_month} Churn Forecast"
)


# ============================================================
# KPIs
# ============================================================

predicted_churn_rate = (
    result_df[
        "Predicted_Churn"
    ].mean()
    * 100
)

average_churn_risk = (
    result_df[
        "Churn_Probability"
    ].mean()
    * 100
)

high_risk_count = (
    result_df[
        "Risk_Segment"
    ]
    .eq("High Risk")
    .sum()
)

medium_risk_count = (
    result_df[
        "Risk_Segment"
    ]
    .eq("Medium Risk")
    .sum()
)

low_risk_count = (
    result_df[
        "Risk_Segment"
    ]
    .eq("Low Risk")
    .sum()
)


col1, col2, col3, col4, col5 = (
    st.columns(5)
)

col1.metric(
    "Predicted Churn",
    f"{predicted_churn_rate:.2f}%"
)

col2.metric(
    "Average Churn Risk",
    f"{average_churn_risk:.2f}%"
)

col3.metric(
    "High Risk",
    f"{high_risk_count:,}"
)

col4.metric(
    "Medium Risk",
    f"{medium_risk_count:,}"
)

col5.metric(
    "Low Risk",
    f"{low_risk_count:,}"
)


# ============================================================
# RISK DISTRIBUTION
# ============================================================

st.subheader(
    "Risk Segment Distribution"
)

risk_counts = (
    result_df[
        "Risk_Segment"
    ]
    .value_counts()
    .reindex(
        [
            "Low Risk",
            "Medium Risk",
            "High Risk"
        ]
    )
    .fillna(0)
)

st.bar_chart(
    risk_counts
)


# ============================================================
# RISK SUMMARY
# ============================================================

st.subheader(
    "Risk Segment Summary"
)

summary = (
    result_df
    .groupby(
        "Risk_Segment",
        observed=True
    )
    .agg(
        Customers=(
            "Churn_Probability",
            "size"
        ),
        Average_Churn_Probability=(
            "Churn_Probability",
            "mean"
        )
    )
    .reset_index()
)

summary[
    "Average_Churn_Probability"
] *= 100

summary = summary.rename(
    columns={
        "Risk_Segment":
            "Risk Segment",
        "Average_Churn_Probability":
            "Average Churn Probability (%)"
    }
)

st.dataframe(
    summary,
    use_container_width=True
)


# ============================================================
# HIGH-RISK CUSTOMERS
# ============================================================

st.subheader(
    "🚨 Customers Requiring Retention Attention"
)

high_risk_df = result_df[
    result_df[
        "Risk_Segment"
    ] == "High Risk"
].copy()


display_columns = [
    col
    for col in [
        "mobile_number",
        "current_month_arpu",
        "total_rech_amt",
        "total_rech_num",
        "total_og_mou",
        "total_ic_mou",
        "vol_2g_mb",
        "vol_3g_mb",
        "Churn_Probability",
        "Risk_Segment"
    ]
    if col in high_risk_df.columns
]

high_risk_display = (
    high_risk_df[
        display_columns
    ].copy()
)

if "Churn_Probability" in high_risk_display:

    high_risk_display[
        "Churn_Probability"
    ] *= 100

high_risk_display = (
    high_risk_display.rename(
        columns={
            "mobile_number":
                "Customer ID",
            "current_month_arpu":
                "ARPU",
            "total_rech_amt":
                "Recharge Amount",
            "total_rech_num":
                "Recharge Count",
            "total_og_mou":
                "Outgoing Usage",
            "total_ic_mou":
                "Incoming Usage",
            "vol_2g_mb":
                "2G Usage MB",
            "vol_3g_mb":
                "3G Usage MB",
            "Churn_Probability":
                "Churn Probability (%)",
            "Risk_Segment":
                "Risk Segment"
        }
    )
)

st.dataframe(
    high_risk_display,
    use_container_width=True
)


# ============================================================
# DOWNLOAD
# ============================================================

st.subheader(
    "Download Predictions"
)

download_csv = (
    result_df
    .to_csv(index=False)
    .encode("utf-8")
)

st.download_button(
    label="⬇️ Download Churn Predictions",
    data=download_csv,
    file_name=(
        f"{month.lower()}_"
        f"to_{next_month.lower()}_"
        "churn_predictions.csv"
    ),
    mime="text/csv"
)


# ============================================================
# BUSINESS INTERPRETATION
# ============================================================

st.divider()

st.header(
    "💡 Business Interpretation"
)

if predicted_churn_rate >= 10:

    st.warning(
        f"Predicted {next_month} churn is "
        f"{predicted_churn_rate:.2f}%. "
        "Prioritize targeted retention campaigns "
        "for high-risk customers."
    )

elif predicted_churn_rate >= 5:

    st.info(
        f"Predicted {next_month} churn is "
        f"{predicted_churn_rate:.2f}%. "
        "Focus retention offers on high- and "
        "medium-risk customers."
    )

else:

    st.success(
        f"Predicted {next_month} churn is "
        f"{predicted_churn_rate:.2f}%. "
        "Overall churn risk is relatively low."
    )

