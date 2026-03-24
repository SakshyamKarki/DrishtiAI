import './App.css'
import Navbar from "./components/Navbar"
import Footer from "./components/Footer"
import {Outlet} from 'react-router-dom'
import Upload from "./components/Upload";
import Result from "./components/Result";

function App() {
  const [result, setResult] = useState(null);

  return (
    <>
      <Navbar/>
      <main>
        <Outlet/>
        <div>
          <h1>DrishtiAI</h1>
          <Upload setResult={setResult} />
          <Result result={result} />
        </div>
      </main>
      <Footer/>
    </>
  )
}

export default App
