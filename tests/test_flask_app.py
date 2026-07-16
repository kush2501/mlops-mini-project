from flask_app.app import app


# -------------------- Test Client -------------------- #

client = app.test_client()


# -------------------- Home Page Test -------------------- #

def test_home_page():
    """
    Test whether home page loads successfully.
    """

    response = client.get("/")

    assert response.status_code == 200


# -------------------- Prediction Test -------------------- #

def test_prediction():
    """
    Test prediction endpoint.
    """

    response = client.post(
        "/predict",
        data={
            "text": "I am very happy today"
        }
    )

    assert response.status_code == 200

    # Check prediction is present
    assert b"Positive" in response.data or b"Negative" in response.data