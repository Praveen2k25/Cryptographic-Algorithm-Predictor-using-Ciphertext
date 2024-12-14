import os
import random
import string
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.backends import default_backend
import joblib
import math


def generate_plaintexts(num_samples, min_length=8, max_length=64):
    characters = string.ascii_letters + string.digits
    plaintexts = []

    for _ in range(num_samples):
        length = random.randint(min_length, max_length)
        plaintext = ''.join(random.choice(characters) for _ in range(length))
        plaintexts.append(plaintext)

    return plaintexts



def encrypt_aes(plaintext, key):
    iv = os.urandom(16)
    cipher = Cipher(algorithms.AES(key), modes.CFB(iv), backend=default_backend())
    encryptor = cipher.encryptor()
    ciphertext = encryptor.update(plaintext.encode('utf-8')) + encryptor.finalize()
    return iv + ciphertext


def encrypt_rsa(plaintext, public_key):
    ciphertext = public_key.encrypt(
        plaintext.encode('utf-8'),
        padding.OAEP(
            mgf=padding.MGF1(algorithm=hashes.SHA256()),
            algorithm=hashes.SHA256(),
            label=None
        )
    )
    return ciphertext


def encrypt_sha256(plaintext):
    digest = hashes.Hash(hashes.SHA256(), backend=default_backend())
    digest.update(plaintext.encode('utf-8'))
    return digest.finalize()



def generate_dataset(num_samples=100000):
    aes_key = os.urandom(32)
    rsa_key = rsa.generate_private_key(public_exponent=65537, key_size=2048, backend=default_backend())
    public_key = rsa_key.public_key()

    plaintexts = generate_plaintexts(num_samples)
    dataset = []

    for plaintext in plaintexts:
        # Encrypt using AES
        ciphertext_aes = encrypt_aes(plaintext, aes_key)
        dataset.append((plaintext, ciphertext_aes.hex(), 'AES'))

        # Encrypt using RSA
        ciphertext_rsa = encrypt_rsa(plaintext, public_key)
        dataset.append((plaintext, ciphertext_rsa.hex(), 'RSA'))

        # Hash using SHA-256
        ciphertext_sha256 = encrypt_sha256(plaintext)
        dataset.append((plaintext, ciphertext_sha256.hex(), 'SHA-256'))

    df = pd.DataFrame(dataset, columns=['Plaintext', 'Ciphertext', 'Algorithm'])
    df.to_csv('cryptography.csv', index=False)
    print("Dataset created and saved as 'cryptography_dataset.csv'")
    return df



def byte_frequency(ciphertext):
    bytes_data = bytes.fromhex(ciphertext)
    freq = [bytes_data.count(i) for i in range(256)]
    return freq


def calculate_entropy(ciphertext):
    bytes_data = bytes.fromhex(ciphertext)
    probability_distribution = [bytes_data.count(i) / len(bytes_data) for i in set(bytes_data)]
    entropy = -sum(p * math.log2(p) for p in probability_distribution if p > 0)
    return entropy


def preprocess_data(df):
    df['ByteFrequency'] = df['Ciphertext'].apply(byte_frequency)
    df['Entropy'] = df['Ciphertext'].apply(calculate_entropy)

    features_df = pd.DataFrame(df['ByteFrequency'].tolist())
    features_df['Entropy'] = df['Entropy']


    features_df.columns = features_df.columns.astype(str)

    le = LabelEncoder()
    df['Algorithm'] = le.fit_transform(df['Algorithm'])

    X = features_df
    y = df['Algorithm']

    return X, y, le


def train_and_evaluate(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    print(f"Accuracy: {accuracy:.2f}")
    print("Classification Report:")
    print(report)

    return model


def save_model(model, le):
    joblib.dump(model, '../cryptographic.pkl')
    joblib.dump(le, '../label.pkl')
    print("Model and Label Encoder saved as 'cryptographic_algorithm_model.pkl' and 'label_encoder.pkl'")



if __name__ == "__main__":
    df = generate_dataset()
    X, y, label_encoder = preprocess_data(df)
    model = train_and_evaluate(X, y)
    save_model(model, label_encoder)
