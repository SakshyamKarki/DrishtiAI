import './App.css'
import Navbar from "./components/Navbar"
import Footer from "./components/Footer"
import {Outlet} from 'react-router-dom'
import './styles/auth.css'

function App() {
  return (
    <div className='relative min-h-screen pt-1' style={{ background: "#060611" }}>
      <div className="auth-orb-1" />
      <div className="auth-grid" />
      <Navbar/>
      <main>
        <Outlet/>
      </main>
      <Footer/>
    </div>
  )
}

export default App
