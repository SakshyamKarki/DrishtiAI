import './App.css'
import Navbar from "./components/Navbar"
import Footer from "./components/Footer"
import {Outlet} from 'react-router-dom'
import Upload from "./components/Upload";
import Result from "./components/Result";
import EyeBackground from './components/EyeBackground'
import EyeBackgroundTwo from './components/EyeBackgroundTwo'

function App() {
  // const [result, setResult] = useState(null);

  return (
    <div className='relative min-h-screen' style={{ background: "#060611" }}>
      {/* <EyeBackground/> */}
      {/* <EyeBackgroundTwo /> */}
      <Navbar/>
      <main>
        <Outlet/>
        {/* <div>
          <h1>DrishtiAI</h1>
          <Upload setResult={setResult} />
          <Result result={result} />
        </div> */}
      </main>
      <Footer/>
    </div>
  )
}

export default App
