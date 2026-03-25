

function Navbar(){
    return (
        <nav className="flex justify-between items-center w-4/5 px-8 py-4 m-auto bg-blue-200/20 rounded-2xl text-mauve-400 font-semibold sticky top-3">
            <h1 className="hover:text-white hover:cursor-pointer">DrishtiAI</h1>
            <div className="flex justify-between items-center w-1/4 px-3">
                <p className="hover:cursor-pointer hover:text-white">Detect</p>
                <p className="hover:cursor-pointer hover:text-white">About</p>
                <p className="hover:cursor-pointer hover:text-white">Technology</p>
            </div>
        </nav>
    )
}

export default Navbar;