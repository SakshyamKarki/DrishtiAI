import App from "./App.jsx";
import Home from "./pages/Home.jsx"
import Detect from "./pages/Detect.jsx"
import About from "./pages/About.jsx"

const routes=[
    {
        path:'/',
        element:<App />,
        children:[
            {index:true, element:<Home />},
            {path:'/detect', element:<Detect />},
            {path:'/about', element:<About />},
        ],
    },
]

export default routes;