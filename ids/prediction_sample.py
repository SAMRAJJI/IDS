"""
Interactive IDS Predictor - Test with custom samples (FIXED VERSION)
"""
import numpy as np
import tensorflow as tf
import pickle
import pandas as pd
from colorama import init, Fore, Style
init(autoreset=True)

class IDSPredictor:
    def __init__(self, model_path='models/best_model.h5'):
        """Load trained model and preprocessors"""
        print("Loading model and preprocessors...")
        self.model = tf.keras.models.load_model(model_path)
        
        # Load preprocessors
        with open('models/label_encoders.pkl', 'rb') as f:
            self.label_encoders = pickle.load(f)
        with open('models/scaler.pkl', 'rb') as f:
            self.scaler = pickle.load(f)
        
        # NSL-KDD feature names (41 features, excluding label and difficulty)
        self.feature_names = [
            'duration', 'protocol_type', 'service', 'flag', 'src_bytes',
            'dst_bytes', 'land', 'wrong_fragment', 'urgent', 'hot',
            'num_failed_logins', 'logged_in', 'num_compromised', 'root_shell',
            'su_attempted', 'num_root', 'num_file_creations', 'num_shells',
            'num_access_files', 'num_outbound_cmds', 'is_host_login',
            'is_guest_login', 'count', 'srv_count', 'serror_rate',
            'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate', 'same_srv_rate',
            'diff_srv_rate', 'srv_diff_host_rate', 'dst_host_count',
            'dst_host_srv_count', 'dst_host_same_srv_rate',
            'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate',
            'dst_host_srv_diff_host_rate', 'dst_host_serror_rate',
            'dst_host_srv_serror_rate', 'dst_host_rerror_rate',
            'dst_host_srv_rerror_rate'
        ]
        
        self.categorical_features = ['protocol_type', 'service', 'flag']
        
        # Get valid values for categorical features
        self.valid_values = {}
        for col in self.categorical_features:
            self.valid_values[col] = list(self.label_encoders[col].classes_)
        
        print(Fore.GREEN + "✓ Model loaded successfully!\n")
        self._show_valid_values()
    
    def _show_valid_values(self):
        """Show valid categorical values"""
        print(Fore.CYAN + "Valid categorical values:")
        print(Fore.CYAN + "-" * 60)
        for feature, values in self.valid_values.items():
            print(f"{Fore.YELLOW}{feature}: {Fore.WHITE}{', '.join(map(str, values[:10]))}" + 
                  (f" ... ({len(values)} total)" if len(values) > 10 else ""))
        print(Fore.CYAN + "-" * 60 + "\n")
    
    def validate_categorical(self, feature_name, value):
        """Validate and fix categorical input"""
        # Convert to string and strip whitespace
        value_str = str(value).strip()
        
        # Try exact match first
        if value_str in self.valid_values[feature_name]:
            return value_str
        
        # Try case-insensitive match
        for valid_value in self.valid_values[feature_name]:
            if value_str.lower() == valid_value.lower():
                print(Fore.YELLOW + f"Note: '{value_str}' converted to '{valid_value}'")
                return valid_value
        
        # If still not found, use default
        default = self.valid_values[feature_name][0]
        print(Fore.RED + f"Warning: '{value_str}' not valid for {feature_name}. Using default: '{default}'")
        print(Fore.CYAN + f"Valid values: {', '.join(map(str, self.valid_values[feature_name][:5]))}")
        return default
    
    def preprocess_sample(self, sample_dict):
        """Preprocess a single sample with validation"""
        # Create DataFrame from input
        df = pd.DataFrame([sample_dict])
        
        # Validate and encode categorical features
        for col in self.categorical_features:
            if col in df.columns:
                # Validate the value
                df[col] = self.validate_categorical(col, df[col].iloc[0])
                # Encode
                df[col] = self.label_encoders[col].transform(df[col].astype(str))
        
        # Ensure correct feature order
        df = df[self.feature_names]
        
        # Scale features
        X_scaled = self.scaler.transform(df)
        
        return X_scaled
    
    def predict(self, sample_dict, threshold=0.5):
        """Predict if traffic is normal or attack"""
        try:
            # Preprocess
            X = self.preprocess_sample(sample_dict)
            
            # Get probability
            probability = self.model.predict(X, verbose=0)[0][0]
            
            # Get prediction
            prediction = 1 if probability >= threshold else 0
            confidence = probability if prediction == 1 else (1 - probability)
            
            return {
                'prediction': 'ATTACK' if prediction == 1 else 'NORMAL',
                'attack_probability': probability,
                'confidence': confidence,
                'risk_level': self.get_risk_level(probability)
            }
        except Exception as e:
            print(Fore.RED + f"\n❌ Error during prediction: {e}")
            return None
    
    def get_risk_level(self, probability):
        """Determine risk level"""
        if probability >= 0.8:
            return 'CRITICAL'
        elif probability >= 0.6:
            return 'HIGH'
        elif probability >= 0.4:
            return 'MEDIUM'
        elif probability >= 0.2:
            return 'LOW'
        else:
            return 'SAFE'
    
    def display_result(self, result):
        """Display prediction result with colors"""
        if result is None:
            return
            
        print("\n" + "="*60)
        print("PREDICTION RESULT")
        print("="*60)
        
        if result['prediction'] == 'ATTACK':
            print(Fore.RED + f"⚠️  PREDICTION: {result['prediction']}")
            print(Fore.RED + f"🎯 Attack Probability: {result['attack_probability']:.2%}")
            print(Fore.RED + f"📊 Confidence: {result['confidence']:.2%}")
            print(Fore.RED + f"🚨 Risk Level: {result['risk_level']}")
        else:
            print(Fore.GREEN + f"✓ PREDICTION: {result['prediction']}")
            print(Fore.GREEN + f"🎯 Attack Probability: {result['attack_probability']:.2%}")
            print(Fore.GREEN + f"📊 Confidence: {result['confidence']:.2%}")
            print(Fore.GREEN + f"🚨 Risk Level: {result['risk_level']}")
        
        print("="*60 + "\n")


def get_sample_from_test_data():
    """Load a random sample from test data"""
    # Load original test data to get a real sample
    test_file = 'data/nsl-kdd/KDDTest+.txt'
    
    columns = [
        'duration', 'protocol_type', 'service', 'flag', 'src_bytes',
        'dst_bytes', 'land', 'wrong_fragment', 'urgent', 'hot',
        'num_failed_logins', 'logged_in', 'num_compromised', 'root_shell',
        'su_attempted', 'num_root', 'num_file_creations', 'num_shells',
        'num_access_files', 'num_outbound_cmds', 'is_host_login',
        'is_guest_login', 'count', 'srv_count', 'serror_rate',
        'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate', 'same_srv_rate',
        'diff_srv_rate', 'srv_diff_host_rate', 'dst_host_count',
        'dst_host_srv_count', 'dst_host_same_srv_rate',
        'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate',
        'dst_host_srv_diff_host_rate', 'dst_host_serror_rate',
        'dst_host_srv_serror_rate', 'dst_host_rerror_rate',
        'dst_host_srv_rerror_rate', 'label', 'difficulty'
    ]
    
    df = pd.read_csv(test_file, names=columns, header=None)
    
    # Get random sample
    sample = df.sample(1).iloc[0]
    actual_label = sample['label']
    
    # Remove label and difficulty
    sample_dict = sample.drop(['label', 'difficulty']).to_dict()
    
    return sample_dict, actual_label


def manual_input_mode(predictor):
    """Manually enter feature values with improved UX"""
    print("\n" + Fore.CYAN + "="*60)
    print(Fore.CYAN + "MANUAL INPUT MODE")
    print(Fore.CYAN + "="*60)
    print("\nEnter network traffic features (press Enter for default):\n")
    
    sample = {}
    
    try:
        # Basic features with validation
        sample['duration'] = int(input("Duration (seconds) [0]: ") or 0)
        
        print(f"\n{Fore.YELLOW}Available protocols: {', '.join(predictor.valid_values['protocol_type'])}")
        protocol = input("Protocol [tcp]: ").strip() or 'tcp'
        sample['protocol_type'] = protocol
        
        print(f"\n{Fore.YELLOW}Common services: http, ftp, smtp, private, domain_u, etc.")
        service = input("Service [http]: ").strip() or 'http'
        sample['service'] = service
        
        print(f"\n{Fore.YELLOW}Common flags: SF, S0, REJ, RSTO, SH, etc.")
        flag = input("Flag [SF]: ").strip() or 'SF'
        sample['flag'] = flag
        
        sample['src_bytes'] = int(input("\nSource bytes [232]: ") or 232)
        sample['dst_bytes'] = int(input("Destination bytes [8153]: ") or 8153)
        
        # Connection features
        print(f"\n{Fore.CYAN}--- Connection Features ---")
        sample['land'] = int(input("Land (same host/port) [0]: ") or 0)
        sample['wrong_fragment'] = int(input("Wrong fragment [0]: ") or 0)
        sample['urgent'] = int(input("Urgent packets [0]: ") or 0)
        sample['hot'] = int(input("Hot indicators [0]: ") or 0)
        
        # Authentication features
        print(f"\n{Fore.CYAN}--- Authentication Features ---")
        sample['num_failed_logins'] = int(input("Failed logins [0]: ") or 0)
        sample['logged_in'] = int(input("Logged in (0/1) [1]: ") or 1)
        sample['num_compromised'] = int(input("Compromised conditions [0]: ") or 0)
        sample['root_shell'] = int(input("Root shell (0/1) [0]: ") or 0)
        sample['su_attempted'] = int(input("SU attempted [0]: ") or 0)
        
        # File/shell features
        print(f"\n{Fore.CYAN}--- File/Shell Features ---")
        sample['num_root'] = int(input("Root accesses [0]: ") or 0)
        sample['num_file_creations'] = int(input("File creations [0]: ") or 0)
        sample['num_shells'] = int(input("Shell prompts [0]: ") or 0)
        sample['num_access_files'] = int(input("Access files [0]: ") or 0)
        sample['num_outbound_cmds'] = int(input("Outbound commands [0]: ") or 0)
        
        # Host features
        print(f"\n{Fore.CYAN}--- Host Features ---")
        sample['is_host_login'] = int(input("Is host login [0]: ") or 0)
        sample['is_guest_login'] = int(input("Is guest login [0]: ") or 0)
        
        # Traffic features
        print(f"\n{Fore.CYAN}--- Traffic Statistics (last 2 seconds) ---")
        sample['count'] = int(input("Connections to same host [2]: ") or 2)
        sample['srv_count'] = int(input("Connections to same service [2]: ") or 2)
        sample['serror_rate'] = float(input("SYN error rate [0.0]: ") or 0.0)
        sample['srv_serror_rate'] = float(input("Service SYN error rate [0.0]: ") or 0.0)
        sample['rerror_rate'] = float(input("REJ error rate [0.0]: ") or 0.0)
        sample['srv_rerror_rate'] = float(input("Service REJ error rate [0.0]: ") or 0.0)
        sample['same_srv_rate'] = float(input("Same service rate [1.0]: ") or 1.0)
        sample['diff_srv_rate'] = float(input("Different service rate [0.0]: ") or 0.0)
        sample['srv_diff_host_rate'] = float(input("Service different host rate [0.0]: ") or 0.0)
        
        # Host-based features
        print(f"\n{Fore.CYAN}--- Host-based Statistics (last 100 connections) ---")
        sample['dst_host_count'] = int(input("Destination host count [255]: ") or 255)
        sample['dst_host_srv_count'] = int(input("Destination host service count [255]: ") or 255)
        sample['dst_host_same_srv_rate'] = float(input("Same service rate [1.0]: ") or 1.0)
        sample['dst_host_diff_srv_rate'] = float(input("Different service rate [0.0]: ") or 0.0)
        sample['dst_host_same_src_port_rate'] = float(input("Same source port rate [1.0]: ") or 1.0)
        sample['dst_host_srv_diff_host_rate'] = float(input("Service different host rate [0.0]: ") or 0.0)
        sample['dst_host_serror_rate'] = float(input("SYN error rate [0.0]: ") or 0.0)
        sample['dst_host_srv_serror_rate'] = float(input("Service SYN error rate [0.0]: ") or 0.0)
        sample['dst_host_rerror_rate'] = float(input("REJ error rate [0.0]: ") or 0.0)
        sample['dst_host_srv_rerror_rate'] = float(input("Service REJ error rate [0.0]: ") or 0.0)
        
        return sample
        
    except ValueError as e:
        print(Fore.RED + f"\n❌ Invalid input: {e}")
        print(Fore.YELLOW + "Using default values for remaining features...")
        
        # Fill remaining with defaults
        defaults = {
            'land': 0, 'wrong_fragment': 0, 'urgent': 0, 'hot': 0,
            'num_failed_logins': 0, 'logged_in': 1, 'num_compromised': 0,
            'root_shell': 0, 'su_attempted': 0, 'num_root': 0,
            'num_file_creations': 0, 'num_shells': 0, 'num_access_files': 0,
            'num_outbound_cmds': 0, 'is_host_login': 0, 'is_guest_login': 0,
            'count': 2, 'srv_count': 2, 'serror_rate': 0.0,
            'srv_serror_rate': 0.0, 'rerror_rate': 0.0, 'srv_rerror_rate': 0.0,
            'same_srv_rate': 1.0, 'diff_srv_rate': 0.0, 'srv_diff_host_rate': 0.0,
            'dst_host_count': 255, 'dst_host_srv_count': 255,
            'dst_host_same_srv_rate': 1.0, 'dst_host_diff_srv_rate': 0.0,
            'dst_host_same_src_port_rate': 1.0, 'dst_host_srv_diff_host_rate': 0.0,
            'dst_host_serror_rate': 0.0, 'dst_host_srv_serror_rate': 0.0,
            'dst_host_rerror_rate': 0.0, 'dst_host_srv_rerror_rate': 0.0
        }
        
        for key, value in defaults.items():
            if key not in sample:
                sample[key] = value
        
        return sample


def quick_input_mode(predictor):
    """Quick input mode - only essential features"""
    print("\n" + Fore.CYAN + "="*60)
    print(Fore.CYAN + "QUICK INPUT MODE (Essential Features Only)")
    print(Fore.CYAN + "="*60)
    
    sample = {}
    
    try:
        # Only ask for key features
        sample['duration'] = int(input("\nDuration (seconds) [0]: ") or 0)
        
        print(f"{Fore.YELLOW}Protocols: tcp, udp, icmp")
        protocol = input("Protocol [tcp]: ").strip() or 'tcp'
        sample['protocol_type'] = protocol
        
        print(f"{Fore.YELLOW}Services: http, ftp, smtp, private, etc.")
        service = input("Service [http]: ").strip() or 'http'
        sample['service'] = service
        
        print(f"{Fore.YELLOW}Flags: SF, S0, REJ, RSTO")
        flag = input("Flag [SF]: ").strip() or 'SF'
        sample['flag'] = flag
        
        sample['src_bytes'] = int(input("Source bytes [232]: ") or 232)
        sample['dst_bytes'] = int(input("Destination bytes [8153]: ") or 8153)
        sample['count'] = int(input("Connection count [2]: ") or 2)
        sample['logged_in'] = int(input("Logged in? (1=yes, 0=no) [1]: ") or 1)
        
        # Fill rest with defaults
        defaults = {
            'land': 0, 'wrong_fragment': 0, 'urgent': 0, 'hot': 0,
            'num_failed_logins': 0, 'num_compromised': 0,
            'root_shell': 0, 'su_attempted': 0, 'num_root': 0,
            'num_file_creations': 0, 'num_shells': 0, 'num_access_files': 0,
            'num_outbound_cmds': 0, 'is_host_login': 0, 'is_guest_login': 0,
            'srv_count': 2, 'serror_rate': 0.0,
            'srv_serror_rate': 0.0, 'rerror_rate': 0.0, 'srv_rerror_rate': 0.0,
            'same_srv_rate': 1.0, 'diff_srv_rate': 0.0, 'srv_diff_host_rate': 0.0,
            'dst_host_count': 255, 'dst_host_srv_count': 255,
            'dst_host_same_srv_rate': 1.0, 'dst_host_diff_srv_rate': 0.0,
            'dst_host_same_src_port_rate': 1.0, 'dst_host_srv_diff_host_rate': 0.0,
            'dst_host_serror_rate': 0.0, 'dst_host_srv_serror_rate': 0.0,
            'dst_host_rerror_rate': 0.0, 'dst_host_srv_rerror_rate': 0.0
        }
        
        for key, value in defaults.items():
            if key not in sample:
                sample[key] = value
        
        return sample
        
    except Exception as e:
        print(Fore.RED + f"Error: {e}")
        return None


def main():
    """Main interactive predictor"""
    print(Fore.CYAN + """
    ╔═══════════════════════════════════════════════════════════╗
    ║                                                           ║
    ║          🛡️  INTRUSION DETECTION SYSTEM 🛡️                ║
    ║              Deep Learning Predictor                      ║
    ║                                                           ║
    ╚═══════════════════════════════════════════════════════════╝
    """)
    
    # Load predictor
    predictor = IDSPredictor()
    
    while True:
        print(Fore.YELLOW + "\n" + "="*60)
        print(Fore.YELLOW + "SELECT MODE:")
        print(Fore.YELLOW + "="*60)
        print("1. Test with random sample from test data")
        print("2. Quick input (only essential features)")
        print("3. Full manual input (all 41 features)")
        print("4. Batch prediction from CSV file")
        print("5. Show valid categorical values")
        print("6. Exit")
        
        choice = input("\nEnter your choice (1-6): ").strip()
        
        if choice == '1':
            # Random sample from test data
            print("\nLoading random sample from test data...")
            sample_dict, actual_label = get_sample_from_test_data()
            
            print("\n" + Fore.CYAN + "Sample Features:")
            print(Fore.CYAN + "-" * 60)
            for key, value in list(sample_dict.items())[:10]:
                print(f"{key}: {value}")
            print("...")
            print(Fore.CYAN + "-" * 60)
            print(f"\n{Fore.YELLOW}Actual Label: {actual_label}")
            
            # Predict
            result = predictor.predict(sample_dict)
            predictor.display_result(result)
            
        elif choice == '2':
            # Quick input
            sample_dict = quick_input_mode(predictor)
            if sample_dict:
                result = predictor.predict(sample_dict)
                predictor.display_result(result)
            
        elif choice == '3':
            # Full manual input
            sample_dict = manual_input_mode(predictor)
            result = predictor.predict(sample_dict)
            predictor.display_result(result)
        
        elif choice == '4':
            # Batch prediction
            csv_file = input("\nEnter CSV file path: ").strip()
            batch_predict(predictor, csv_file)
        
        elif choice == '5':
            # Show valid values
            predictor._show_valid_values()
            
        elif choice == '6':
            print(Fore.GREEN + "\n👋 Thank you for using IDS Predictor!")
            break
        else:
            print(Fore.RED + "\n❌ Invalid choice! Please select 1-6.")


def batch_predict(predictor, csv_file):
    """Predict multiple samples from CSV"""
    try:
        print(f"\nLoading data from {csv_file}...")
        df = pd.read_csv(csv_file)
        
        print(f"Found {len(df)} samples to predict.\n")
        
        results = []
        for idx, row in df.iterrows():
            sample_dict = row.to_dict()
            result = predictor.predict(sample_dict)
            if result:
                results.append({
                    'Sample_ID': idx + 1,
                    'Prediction': result['prediction'],
                    'Attack_Probability': f"{result['attack_probability']:.2%}",
                    'Confidence': f"{result['confidence']:.2%}",
                    'Risk_Level': result['risk_level']
                })
            
            # Show progress
            if (idx + 1) % 100 == 0:
                print(f"Processed {idx + 1}/{len(df)} samples...")
        
        # Save results
        results_df = pd.DataFrame(results)
        output_file = 'results/batch_predictions.csv'
        results_df.to_csv(output_file, index=False)
        
        print(Fore.GREEN + f"\n✓ Predictions saved to {output_file}")
        
        # Show summary
        print("\n" + "="*60)
        print("PREDICTION SUMMARY")
        print("="*60)
        attack_count = len([r for r in results if r['Prediction'] == 'ATTACK'])
        normal_count = len(results) - attack_count
        print(f"Total Samples: {len(results)}")
        print(f"Normal: {normal_count} ({normal_count/len(results)*100:.1f}%)")
        print(f"Attack: {attack_count} ({attack_count/len(results)*100:.1f}%)")
        print("="*60)
        
    except Exception as e:
        print(Fore.RED + f"\n❌ Error: {e}")


if __name__ == "__main__":
    # Install colorama if needed
    try:
        from colorama import init, Fore, Style
    except ImportError:
        print("Installing colorama for colored output...")
        import subprocess
        subprocess.check_call(['pip', 'install', 'colorama'])
        from colorama import init, Fore, Style
    
    main()