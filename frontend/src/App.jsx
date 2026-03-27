import './App.css'
import Navbar from "./components/Navbar"
import Footer from "./components/Footer"
import {Outlet} from 'react-router-dom'
import './styles/auth.css'

function App() {
  // const [result, setResult] = useState(null);

  return (
    <div className='relative min-h-screen pt-1' style={{ background: "#060611" }}>
      {/* <EyeBackground/> */}
      {/* <EyeBackgroundTwo /> */}
      <div className="auth-orb-1" />
      {/* <div className="auth-orb-2" /> */}
      <div className="auth-grid" />

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
