import sys
import os

# รวมการเรียกใช้จาก dashboard.ml.train
from dashboard.ml.train import train, make_data, run_single_train

if __name__ == '__main__':
    print("=== Training Perceptron with Manual Weight Update ===")
    model, lr_results = train(num_epochs=200, main_lr=0.1)
    print("\n--- 3 Learning Rates Comparison Results ---")
    for lr_str, res in lr_results.items():
        print(f"Learning Rate: {res['lr']:<5} | Final Loss: {res['final_loss']:.4f} | Final Accuracy: {res['final_acc']:.2f}%")
