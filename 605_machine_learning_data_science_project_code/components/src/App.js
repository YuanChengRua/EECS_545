import './App.css';
import React from 'react';

const DROPDOWN_API_ENDPOINT = 'https://mie7o7ybx0.execute-api.us-east-1.amazonaws.com/prod/'; // TODO
const ML_API_ENDPOINT = 'https://hsunhc8zyc.execute-api.us-east-1.amazonaws.com/prod/'; // TODO

// atob is deprecated but this function converts base64string to text string
const decodeFileBase64 = (base64String) => {
  // From Bytestream to Percent-encoding to Original string
  return decodeURIComponent(
    atob(base64String).split("").map(function (c) {
      return "%" + ("00" + c.charCodeAt(0).toString(16)).slice(-2);
    }).join("")
  );
};


function App() {
  const [inputFileData, setInputFileData] = React.useState(''); // represented as bytes data (string)
  const [outputFileData, setOutputFileData] = React.useState(''); // represented as readable data (text string)
  const [inputImage, setInputImage] = React.useState(''); // represented as bytes data (string)
  const [buttonDisable, setButtonDisable] = React.useState(true);
  const [submitButtonText, setSubmitButtonText] = React.useState('Submit');
  const [fileButtonText, setFileButtonText] = React.useState('Upload File');
  const [demoDropdownFiles, setDemoDropdownFiles] = React.useState([]);
  const [selectedDropdownFile, setSelectedDropdownFile] = React.useState('');
  const [OutputFileDatarmse, setOutputFileDatarmse] = React.useState('');

  React.useEffect(() => {
    fetch(DROPDOWN_API_ENDPOINT)
    .then(response => response.json())
    .then(data => {
      // GET request error
      if (data.statusCode === 400) {
        console.log('Sorry! There was an error, the demo files are currently unavailable.')
      }

      // GET request success
      else {
        const s3BucketFiles = JSON.parse(data.body);
        setDemoDropdownFiles(s3BucketFiles["s3Files"]);
      }
    });
  }, [])
  
  // convert file to bytes data
  const convertFileToBytes = (inputFile) => {
    console.log('converting file to bytes...');
    return new Promise((resolve, reject) => {
      const fileReader = new FileReader();
      fileReader.readAsDataURL(inputFile); // reads file as bytes data

      fileReader.onload = () => {
        resolve(fileReader.result);
      };

      fileReader.onerror = (error) => {
        reject(error);
      };
    });
  }
  


  // handle file input
  const handleChange = async (event) => {
    // Clear output text.
    setOutputFileData("");

    console.log('newly uploaded file');
    const inputFile = event.target.value;
    console.log(inputFile);

    // convert file to bytes data
//     const base64Data = await convertFileToBytes(inputFile);
//     const base64DataArray = base64Data.split('base64,'); // need to get rid of 'data:image/png;base64,' at the beginning of encoded string
//     const encodedString = base64DataArray[1];
    setInputFileData(inputFile);
    console.log('file converted successfully');

    // enable submit button
    setButtonDisable(false);
  }

  // handle file submission
  const handleSubmit = (event) => {
    event.preventDefault();

    // temporarily disable submit button
    setButtonDisable(true);
    setSubmitButtonText('Loading Result');

    // make POST request
    console.log('making POST request...');
    fetch(ML_API_ENDPOINT, {
      method: 'POST',
      headers: { "Content-Type": "application/json", "Accept": "text/plain" },
      body: JSON.stringify({ "csv": inputFileData })
    }).then(response => response.json())
    .then(data => {
      console.log('getting response...')
      console.log(data);

      // POST request error
      if (data.statusCode === 400) {
        const outputErrorMessage = JSON.parse(data.errorMessage)['outputResultsData'];
        setOutputFileData(outputErrorMessage);
      }

      // POST request success
      else {
            const outputBytesData = JSON.parse(data.body)['outputResultsData'];
            setOutputFileData(outputBytesData);
            const outputBytesDatarmse = JSON.parse(data.body)['outputResultsDatarmse'];
            setOutputFileDatarmse(decodeFileBase64(outputBytesDatarmse))
      }

      // re-enable submit button
      setButtonDisable(false);
      setSubmitButtonText('Submit');
    })
    .then(() => {
      console.log('POST request success');
    })
  }
  const handleDropdown = (event) => {
    setSelectedDropdownFile(event.target.value);

    // temporarily disable submit button
    setButtonDisable(true);
    setSubmitButtonText('Loading Demo File...');

    // only make POST request on file selection
    if (event.target.value) {
      fetch(DROPDOWN_API_ENDPOINT, {
        method: 'POST',
        body: JSON.stringify({ "fileName": event.target.value })
      }).then(response => response.json())
      .then(data => {

        // POST request error
        if (data.statusCode === 400) {
          console.log('Uh oh! There was an error retrieving the dropdown file from the S3 bucket.')
        }

        // POST request success
        else {
          const dropdownFileBytesData = JSON.parse(data.body)['bytesData'];
          console.log(decodeFileBase64(dropdownFileBytesData));
          setInputFileData(decodeFileBase64(dropdownFileBytesData));
          setInputImage('data:image/png;base64,' + dropdownFileBytesData); // hacky way of setting image from bytes data - even works on .jpeg lol
          setSubmitButtonText('Submit');
          setButtonDisable(false);
        }
      });
    }

    else {
      setInputFileData('');
    }
  }

  return (
    <div className="App">
      <div className="Input">
        <h1>Yuan Cheng Final Project</h1>
        <a href="https://github.com/YuanChengRua/eecs-605-docker/blob/main/components/Technical%20Report.pdf">Click here to check the technical report</a>
        <a href="https://github.com/YuanChengRua/EECS-605-project/blob/main/final_project.mp4">Click here to check the presentation video</a>
        <h2>Please select the stored examples here</h2>
        <label htmlFor="demo-dropdown">Demo: </label>
        <select name="Select Image" id="demo-dropdown" value={selectedDropdownFile} onChange={handleDropdown}>
            <option value="">-- Select Demo File --</option>
            {demoDropdownFiles.map((file) => <option key={file} value={file}>{file}</option>)}
        </select>
        <form onSubmit={handleSubmit}>
          <label htmlFor="file-upload">{fileButtonText}</label>
          <h2>Please enter your interested stock name</h2>
          <input type="text" onChange={handleChange} />
          <button type="submit" disabled={buttonDisable}>{submitButtonText}</button>
        </form>
        <img src={`data:;base64,${outputFileData}`} alt="" />
      </div>
      <div className="Output">
        <h1>Results</h1>
        <p>{OutputFileDatarmse}</p>
      </div>
    </div>
  );
}



export default App;
