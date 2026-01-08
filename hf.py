from huggingface_hub import upload_folder

upload_folder(
    folder_path=r"C:\Users\Alie\Downloads\tksprktk\bert_output_final\bert_output_final",
    repo_id="fatahm0987/indobert-spam-sms",
    repo_type="model"
)
