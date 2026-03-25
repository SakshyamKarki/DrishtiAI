import './App.css'
import Navbar from "./components/Navbar"
import Footer from "./components/Footer"
import {Outlet} from 'react-router-dom'
import EyeBackground from './components/EyeBackground'
import EyeBackgroundTwo from './components/EyeBackgroundTwo'

function App() {

  return (
    <div className='relative bg-[#00022d] text-white h-[1000px] '>
      {/* <EyeBackground/> */}
      <EyeBackgroundTwo />
      <Navbar/>
      <main>
        <Outlet/>
      </main>
      <Footer/>
    </div>
  )
}

export default App
