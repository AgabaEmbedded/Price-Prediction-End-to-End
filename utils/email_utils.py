import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import ssl
import os
from dotenv import load_dotenv

load_dotenv()
APP_PASSWORD = os.getenv("APP_PASSWORD")

def configure_and_send_email(sender_email, sender_password, recipient_email, subject, body):
    try:
        # Create the email message
        message = MIMEMultipart()
        message["From"] = sender_email
        message["To"] = recipient_email
        message["Subject"] = subject

        # Attach the email body
        message.attach(MIMEText(body, "plain"))

        # Secure connection with SSL
        context = ssl.create_default_context()

        # Connect to Gmail's SMTP server
        with smtplib.SMTP_SSL("smtp.gmail.com", 465, context=context) as server:
            server.login(sender_email, sender_password)
            server.sendmail(sender_email, recipient_email, message.as_string())

        print("✅ Email sent successfully!")

    except smtplib.SMTPAuthenticationError:
        print("❌ Authentication failed. Check your email/password or enable app passwords.")
    except Exception as e:
        print(f"❌ Failed to send email: {e}")

# Example usage
def send_email(message: str):
    # Replace with your details
    sender = "sundayabraham357@gmail.com"
    password = APP_PASSWORD  # Use an App Password, not your main password
    recipient = "sundayabraham025@gmail.com"
    subject = "Daily Signal"
    body = message

    configure_and_send_email(sender, password, recipient, subject, body)
