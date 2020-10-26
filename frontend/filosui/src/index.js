import React from 'react';
import ReactDOM from 'react-dom';
import 'fontsource-roboto';
import './index.css'
import Dashboard from "./components/Dashboard";
import { MemoryRouter } from "react-router-dom"


class FilosApp extends React.Component {
    render() {
        return (<Dashboard/>);
    }
}

ReactDOM.render(
    <MemoryRouter>
        <FilosApp/>
    </MemoryRouter>,
    document.getElementById('root')
);
