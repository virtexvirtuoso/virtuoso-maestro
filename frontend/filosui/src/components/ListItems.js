import React, {Component} from 'react';
import ListItem from '@material-ui/core/ListItem';
import ListItemIcon from '@material-ui/core/ListItemIcon';
import ListItemText from '@material-ui/core/ListItemText';
import DashboardIcon from '@material-ui/icons/Dashboard';
import AssessmentIcon from '@material-ui/icons/Assessment';
import TrendingUpIcon from '@material-ui/icons/TrendingUp';
import {withRouter} from "react-router-dom";
import List from "@material-ui/core/List";

class MainListItems extends Component {

    constructor(props) {
        super(props);
        this.state = {
            selected: '/'
        }
    }


    updateSelected(path) {
        this.props.history.push(path)
        this.setState({selected: path})
    }

    componentDidUpdate(prevProps, prevState, snapshot) {
        let path = "/" + this.props.location.pathname.split("/")[1];
        if (path !== this.state.selected) {
            this.setState({selected: path})
        }
    }

    render() {
        const selected  = this.state.selected;

        return (
            <List>
                <ListItem button
                          onClick={() => this.updateSelected('/')}
                          selected={selected === '/'}>
                    <ListItemIcon>
                        <DashboardIcon />
                    </ListItemIcon>
                    <ListItemText primary="Home" />
                </ListItem>
                <ListItem button
                          onClick={() => this.updateSelected('/results')}
                          selected={selected === '/results'}>
                    <ListItemIcon>
                        <AssessmentIcon />
                    </ListItemIcon>
                    <ListItemText primary="Results" />
                </ListItem>
                <ListItem button
                          onClick={() => this.updateSelected('/evaluate')}
                          selected={selected === '/evaluate'}>
                    <ListItemIcon>
                        <TrendingUpIcon />
                    </ListItemIcon>
                    <ListItemText primary="Evaluation" />
                </ListItem>
            </List>
        );
    }
}

export default withRouter(MainListItems);
