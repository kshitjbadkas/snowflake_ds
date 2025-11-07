from snowflake.ml.feature_store import FeatureStore, Entity, FeatureView, CreationMode
from snowflake.snowpark.functions import col
import pandas as pd

# 1️⃣ Feature Store Initialization
FS_DB = "ML_ASSIGNMENT_DB"
FS_SCHEMA = "FEATURE_STORE_SCHEMA"

fs = FeatureStore(
    session=session,
    database=FS_DB,
    name=FS_SCHEMA,
    default_warehouse="MY_WH",
    creation_mode=CreationMode.CREATE_IF_NOT_EXIST
)
print("✅ Feature Store Initialized.")

# 2️⃣ Register Entity
customer_entity = Entity(
    name="CUSTOMER",
    join_keys=["CUSTOMER_ID"]
)
fs.register_entity(customer_entity)
print("✅ Entity 'CUSTOMER' Registered.")

# 3️⃣ Define & Register Feature View
feature_df = session.table("CUSTOMER_AGGREGATE_FEATURES")

customer_features_fv = FeatureView(
    name="CUSTOMER_AGGREGATE_FEATURES_FV",
    entities=[customer_entity],
    feature_df=feature_df,
    timestamp_col="EVENT_TIMESTAMP"
)

# ✅ Register Feature View (remove mode)
fs.register_feature_view(
    customer_features_fv,
    version="v1"
)
print("✅ Feature View 'CUSTOMER_AGGREGATE_FEATURES_FV' Registered.")

# 4️⃣ Create Spine DataFrame
spine_data = {
    "CUSTOMER_ID": [101, 102, 103],
    "REQUEST_TIMESTAMP": [
        pd.to_datetime("2023-10-06 00:00:00"),
        pd.to_datetime("2023-10-16 00:00:00"),
        pd.to_datetime("2023-10-10 12:00:00")
    ],
    "LABEL": [0.1, 0.9, 0.5]
}
spine_df = session.create_dataframe(pd.DataFrame(spine_data))
print("✅ Spine DataFrame Created.")

# 5️⃣ Retrieve Feature View
customer_features_fv = fs.get_feature_view(
    name="CUSTOMER_AGGREGATE_FEATURES_FV",
    version="v1"
)

# 6️⃣ Retrieve Features for Training
training_sdf = fs.retrieve_feature_values(
    spine_df=spine_df,
    features=[customer_features_fv],
    spine_timestamp_col="REQUEST_TIMESTAMP"
)

print("\n--- Retrieved Training Dataset ---")
training_sdf.show()

# 7️⃣ Train Model
from snowflake.ml.modeling.xgboost import XGBRegressor

FEATURE_COLS = ["TOTAL_LIFETIME_SPEND", "TOTAL_ORDERS", "AVG_ORDER_VALUE"]
LABEL_COL = "LABEL"

regressor = XGBRegressor(
    input_cols=FEATURE_COLS,
    label_cols=LABEL_COL,
    output_cols="PREDICTION"
)

regressor.fit(training_sdf)
print("\n✅ Model Trained Successfully.")

# 8️⃣ Predict
predictions_sdf = regressor.predict(training_sdf.drop(col(LABEL_COL)))

print("\n--- Prediction Results ---")
predictions_sdf.select("CUSTOMER_ID", *FEATURE_COLS, "PREDICTION").show()
