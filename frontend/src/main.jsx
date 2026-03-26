import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import {Provider} from 'react-redux'
import {ToastContainer} from 'react-toastify'
import './index.css'
import {RouterProvider, createBrowserRouter} from 'react-router-dom'
// import routes from './routes.jsx'
import routes from './routes/routes.jsx'
import "react-toastify/dist/ReactToastify.css"
import store from './app/store.js'
import App from './App.jsx'

const router=createBrowserRouter(routes);

createRoot(document.getElementById('root')).render(
  <StrictMode>
    <Provider store={store}>
    <RouterProvider router={router}/>
    <ToastContainer 
      position='top-right'
      autoClose={3000}
      hideProgressBar={false}
      pauseOnHover={true}
    />
    </Provider>
  </StrictMode>,
);
