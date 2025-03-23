# Complaint Detection UI

A beautiful and user-friendly interface for testing the complaint detection API.

## Features

- Authenticate with token-based authentication
- Analyze conversations in chunks of 4 utterances (configurable)
- Display detection results for each chunk
- Calculate and visualize the percentage of complaints in the whole conversation
- Real-time charts to visualize the complaint trend in the conversation
- Multiple visualization options (line chart, bar chart, comparison chart)
- Beautiful and responsive UI built with Material UI
- Sample conversations for quick testing

## Screenshots

(Screenshots would be added here once the application is deployed)

## Project Structure

```
ui/
├── package.json           # Node dependencies
├── public/                # Static files
└── src/                   # Source code
    ├── components/        # React components
    │   ├── Analysis/      # Analysis visualization components
    │   ├── Auth/          # Authentication components
    │   ├── Conversation/  # Conversation components
    │   ├── Dashboard/     # Main dashboard
    │   └── Layout/        # Layout components
    ├── services/          # API services
    ├── utils/             # Utility functions
    ├── App.js             # Main App component
    ├── index.js           # Entry point
    └── index.css          # Global styles
```

## Installation

1. Make sure the API is running on http://localhost:8000
2. Install dependencies:

```bash
cd ui
npm install
```

3. Start the development server:

```bash
npm start
```

The UI will be available at http://localhost:3000

## Usage

1. Log in using the default credentials (admin/adminpassword)
2. Enter a conversation text with the format "Speaker: Text" on each line
3. Choose the chunk size (default is 4 utterances per chunk)
4. Click "Analyze Conversation"
5. View the analysis results:
   - Summary statistics
   - Complaint trend visualization
   - Detailed analysis of each chunk

You can also upload a conversation text file or select from the sample conversations.

## Configuration

The application uses the following environment variables:

- `REACT_APP_API_URL`: The URL of the API (default: http://localhost:8000)

You can create a `.env` file in the root directory to set these variables:

```
REACT_APP_API_URL=http://localhost:8000
```

## Technical Details

- **React**: Front-end library for building user interfaces
- **Material UI**: Component library for styled UI components
- **Chart.js**: Charting library for visualizations
- **Axios**: HTTP client for API requests
- **React Router**: For routing (if needed in future updates)

## Future Improvements

- Add support for real-time analysis
- Implement user management
- Add conversation history
- Support for different languages
- Export analysis results to PDF or CSV
- Implement dark mode
- Add more visualization options 