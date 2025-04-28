# Install Redis
sudo apt-get install redis-server

# Install dependencies
pip install -r requirements_enhanced.txt

# Start Redis
sudo service redis-server start

# Start the server
python enhanced_main.py