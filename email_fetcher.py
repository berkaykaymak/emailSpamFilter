import imaplib
import email
from email.header import decode_header


def get_latest_email_content(username, password):
    """
    Connects to Gmail, retrieves the latest email, and decodes it.
    Returns: A string containing the Subject and Body of the email.
    """
    imap_url = "imap.gmail.com"

    try:
        # Connect to SSL IMAP
        mail = imaplib.IMAP4_SSL(imap_url)

        # Login
        mail.login(username, password)

        # Select 'inbox'
        mail.select("inbox")

        # Search for all emails
        status, messages = mail.search(None, "ALL")

        # Get the ID of the latest email
        mail_ids = messages[0].split()
        if not mail_ids:
            return None

        latest_email_id = mail_ids[-3]

        # Fetch the email data (RFC822 format)
        status, msg_data = mail.fetch(latest_email_id, "(RFC822)")

        for response_part in msg_data:
            if isinstance(response_part, tuple):
                msg = email.message_from_bytes(response_part[1])

                # Decode Subject
                subject, encoding = decode_header(msg["Subject"])[0]
                if isinstance(subject, bytes):
                    subject = subject.decode(encoding if encoding else "utf-8")

                # Decode Body
                body = ""
                if msg.is_multipart():
                    for part in msg.walk():
                        content_type = part.get_content_type()
                        content_disposition = str(part.get("Content-Disposition"))

                        # We only want text/plain content, skipping attachments
                        if content_type == "text/plain" and "attachment" not in content_disposition:
                            try:
                                body = part.get_payload(decode=True).decode()
                            except:
                                pass  # Skip if decoding fails
                else:
                    # If email is not multipart (simple text)
                    try:
                        body = msg.get_payload(decode=True).decode()
                    except:
                        pass

                # Logout and close
                mail.close()
                mail.logout()

                # Return combined text
                return f"Subject: {subject}\nBody: {body}"

    except Exception as e:
        print(f"Error fetching email: {e}")
        return None