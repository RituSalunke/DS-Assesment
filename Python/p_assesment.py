import streamlit as st
import pandas as pd

st.set_page_config(page_title="Fruit Store Console", layout="centered")

st.title('🍎 Fruit Store Console Application ✨')

# Initialize session state
if 'df' not in st.session_state:
    st.session_state.df = pd.DataFrame({
        'Fruits': ['Apple', 'Banana', 'Orange', 'Strawberry', 'Watermelon', 'Grapes', 'Mango'],
        'Qty': [15, 13, 14, 16, 18, 12, 13],
        'Price': [20, 22, 14, 31, 21, 22, 10]
    })

df = st.session_state.df
fruits = df['Fruits'].tolist()

# Main Role Selection
role = st.selectbox("Select Role", ('Fruit Manager', 'Customer'))

# Fruit Manager Module
if role == 'Fruit Manager':
    action = st.selectbox("Select Action", ('Add Fruit Stock', 'View Fruit Stock', 'Update Fruit Stock'))

    if action == 'Add Fruit Stock':
        st.subheader("➕ Add Fruit Stock")
        col1, col2, col3 = st.columns([2, 1, 1])
        selected_fruit = col1.selectbox('Select Fruit', fruits)
        add_qty = col2.number_input('Enter Quantity', min_value=0, step=1)
        add_price = col3.number_input('Enter Price', min_value=0, step=1)

        if st.button('Add'):
            index = df[df['Fruits'] == selected_fruit].index[0]
            df.at[index, 'Qty'] += add_qty
            df.at[index, 'Price'] += add_price  # Assuming latest price is updated
            st.success(f"{add_qty} units of {selected_fruit} added successfully.")

    elif action == 'View Fruit Stock':
        st.subheader("📦 Current Fruit Stock")
        st.dataframe(df, use_container_width=True)

    elif action == 'Update Fruit Stock':
        st.subheader("🔁 Update Fruit Stock")
        selected_fruit = st.selectbox('Select Fruit to Update', fruits)
        new_qty = st.number_input('Enter New Quantity', min_value=0, step=1)
        new_price = st.number_input('Enter New Price', min_value=0, step=1)

        if st.button('Update'):
            index = df[df['Fruits'] == selected_fruit].index[0]
            df.at[index, 'Qty'] = new_qty
            df.at[index, 'Price'] = new_price
            st.success(f"{selected_fruit} updated successfully.")

# Customer Module
elif role == 'Customer':
    action = st.selectbox("Select Action", ('View Fruit Stock', 'Buy Fruit'))

    if action == 'View Fruit Stock':
        st.subheader("🛒 Available Fruits")
        st.dataframe(df, use_container_width=True)

    elif action == 'Buy Fruit':
        st.subheader("🧾 Buy Fruits")
        selected_fruit = st.selectbox('Select Fruit', fruits)
        buy_qty = st.number_input('Enter Quantity', min_value=1, step=1)

        index = df[df['Fruits'] == selected_fruit].index[0]
        available_qty = df.at[index, 'Qty']
        fruit_price = df.at[index, 'Price']

        if st.button("Buy"):
            if buy_qty <= available_qty:
                df.at[index, 'Qty'] -= buy_qty
                total_cost = buy_qty * fruit_price
                st.success(f"Successfully bought {buy_qty} {selected_fruit}(s) for ₹{total_cost}")
            else:
                st.error(f"Only {available_qty} units available in stock.")
